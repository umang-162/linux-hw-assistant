"""LLM-facing tools: JSON schemas, inspectors and a safe dispatcher.

The LLM may only *name* one of the seven tools below. Tools take no
arguments that influence what runs, so there is nothing for a prompt
injection to smuggle into a command line. Each inspector:

1. asks :mod:`src.executor` for allow-listed commands / files,
2. falls back gracefully (missing binary → /proc or /sys; no root → partial
   data plus an explicit ``status``),
3. normalises output with :mod:`src.parsers`,
4. returns a compact JSON envelope with ``status`` and ``unknown_fields`` so
   the model can say "I don't know" instead of guessing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import parsers as P
from .executor import AccessLog, SystemSource, detect_virtualization

log = logging.getLogger(__name__)

MAX_PAYLOAD_CHARS = 12000

# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #


@dataclass
class ToolOutcome:
    name: str
    payload: dict[str, Any]
    accessed: list[str] = field(default_factory=list)
    cached: bool = False
    duration_ms: int = 0

    @property
    def status(self) -> str:
        return str(self.payload.get("status", "ok"))

    def to_message_content(self) -> str:
        return json.dumps(self.payload, separators=(",", ":"), ensure_ascii=False, default=str)


class ToolContext:
    """Per-session state shared by inspectors (source + cached VM detection)."""

    def __init__(self, source: SystemSource, physical_ram_slots: int | None = None) -> None:
        self.source = source
        self.physical_ram_slots = physical_ram_slots
        self._virt: dict[str, Any] | None = None
        self._virt_log: list[str] = []

    def virtualization(self, access: AccessLog) -> dict[str, Any]:
        if self._virt is None:
            tmp = AccessLog()
            self._virt = detect_virtualization(self.source, tmp)
            self._virt_log = tmp.entries
            access.entries.extend(tmp.entries)
        return self._virt

    @property
    def data_source(self) -> str:
        if self.source.name == "mock":
            return f"mock:{getattr(self.source, 'scenario', 'custom')}"
        return "live"

    # convenience wrappers that also record what was touched
    def run(self, key: str, access: AccessLog):
        res = self.source.run(key)
        access.add(res.summary())
        return res

    def read(self, path: str, access: AccessLog, record: bool = True) -> str | None:
        value = self.source.read_value(path)
        if record:
            access.add(f"read {path} → {'ok' if value is not None else 'unavailable'}")
        return value

    def read_full(self, path: str, access: AccessLog) -> str | None:
        res = self.source.read_text(path)
        access.add(f"read {path} → {'ok' if res.ok else res.error_kind}")
        return res.content if res.ok else None


def _envelope(ctx: ToolContext, tool: str, access: AccessLog, data: dict[str, Any],
              status: str = "ok", message: str | None = None,
              unknown_fields: list[str] | None = None,
              notes: list[str] | None = None) -> dict[str, Any]:
    virt = ctx.virtualization(access)
    env: dict[str, Any] = {
        "tool": tool,
        "status": status,
        "data_source": ctx.data_source,
        "virtualized_environment": virt["virtualized"],
    }
    if virt["virtualized"]:
        env["virtualization_type"] = virt["type"]
    if message:
        env["message"] = message
    if unknown_fields:
        env["unknown_fields"] = unknown_fields
    env["data"] = data
    if notes:
        env["notes"] = notes
    return env


def _dmi_id_info(ctx: ToolContext, access: AccessLog) -> dict[str, Any]:
    base = "/sys/devices/virtual/dmi/id/"
    names = ("sys_vendor", "product_name", "product_version", "product_family", "board_vendor",
             "board_name", "bios_vendor", "bios_version", "bios_date", "chassis_type")
    if not ctx.source.exists(base):
        access.add(f"read {base} → unavailable")
        return {}
    values = {n: ctx.source.read_value(base + n) for n in names}
    access.add(f"read {base}{{{','.join(names[:3])},…}} → ok")
    return {k: v for k, v in P.parse_dmi_id(values).items() if v}


# --------------------------------------------------------------------------- #
# Inspectors
# --------------------------------------------------------------------------- #

def inspect_cpu(ctx: ToolContext, access: AccessLog) -> dict[str, Any]:
    notes: list[str] = []
    data: dict[str, Any] = {}
    method = None

    res = ctx.run("lscpu", access)
    if res.ok:
        data = P.parse_lscpu(res.stdout)
        method = "lscpu"
        topo_res = ctx.run("lscpu_topology", access)
        if topo_res.ok:
            topo = P.parse_lscpu_topology(topo_res.stdout)
            if topo:
                data["physical_cores"] = topo["physical_cores"]
                data["logical_cpus"] = topo["logical_cpus"]
                data["cores_with_smt"] = topo["cores_with_smt"]
                data["cores_without_smt"] = topo["cores_without_smt"]
                data["hybrid_core_layout"] = topo["hybrid_core_layout"]
                if topo["hybrid_core_layout"]:
                    notes.append(
                        "Hybrid layout: some cores have 2 threads (SMT) and some have 1. "
                        "Use logical_cpus for the true thread count; 'threads_per_core_reported' "
                        "from lscpu is only the maximum.")
    if not data or not data.get("model_name"):
        cpuinfo = ctx.read_full("/proc/cpuinfo", access)
        if cpuinfo:
            fallback = P.parse_proc_cpuinfo(cpuinfo)
            if fallback:
                method = method or "/proc/cpuinfo (lscpu unavailable)"
                for k, v in fallback.items():
                    if data.get(k) in (None, [], {}):
                        data[k] = v

    # cpufreq sysfs: max/min frequency (fallback) + governor (always)
    cpu_dir = "/sys/devices/system/cpu/"
    cpus = [d for d in ctx.source.list_dir(cpu_dir) if re.fullmatch(r"cpu\d+", d)]
    if cpus:
        max_khz = [P.to_int(ctx.source.read_value(f"{cpu_dir}{c}/cpufreq/cpuinfo_max_freq"))
                   for c in cpus]
        max_khz = [k for k in max_khz if k]
        if max_khz:
            if data.get("max_mhz") is None:
                data["max_mhz"] = max(max_khz) / 1000
            if len(set(max_khz)) > 1:
                data["max_mhz_per_core_type"] = sorted({k / 1000 for k in max_khz}, reverse=True)
        gov = ctx.source.read_value(f"{cpu_dir}cpu0/cpufreq/scaling_governor")
        drv = ctx.source.read_value(f"{cpu_dir}cpu0/cpufreq/scaling_driver")
        if gov:
            data["frequency_governor"] = gov
        if drv:
            data["frequency_driver"] = drv
        access.add(f"read {cpu_dir}cpu*/cpufreq/* → {'ok' if max_khz or gov else 'unavailable'}")

    if not data:
        return _envelope(ctx, "inspect_cpu", access, {}, status="unavailable",
                         message="Neither lscpu nor /proc/cpuinfo could be read.",
                         unknown_fields=["model_name", "physical_cores", "logical_cpus",
                                         "max_mhz", "caches"])

    data["method"] = method
    unknown = [k for k in ("model_name", "physical_cores", "logical_cpus", "max_mhz")
               if data.get(k) is None]
    virt = ctx.virtualization(access)
    if virt["virtualized"]:
        notes.append("Running in a VM/container: core/thread counts are the virtual CPUs "
                     "assigned to this guest, not necessarily the host's physical CPU.")
    data = {k: v for k, v in data.items() if v not in (None, [], {})}
    return _envelope(ctx, "inspect_cpu", access, data, unknown_fields=unknown or None,
                     notes=notes or None)


def inspect_memory_usage(ctx: ToolContext, access: AccessLog) -> dict[str, Any]:
    res = ctx.run("free", access)
    data = P.parse_free(res.stdout) if res.ok else {}
    method = "free -b"
    if not data:
        meminfo = ctx.read_full("/proc/meminfo", access)
        data = P.parse_meminfo(meminfo) if meminfo else {}
        method = "/proc/meminfo"
    if not data:
        return _envelope(ctx, "inspect_memory_usage", access, {}, status="unavailable",
                         message="Could not read memory usage.",
                         unknown_fields=["ram", "swap"])
    data["method"] = method
    notes = ["ram.total is the memory visible to the OS. It is normally a little lower than "
             "the installed physical RAM because firmware and integrated graphics reserve "
             "some. For installed modules/slots use inspect_ram_slots_and_capacity."]
    return _envelope(ctx, "inspect_memory_usage", access, data, notes=notes)


def _sudoers_hint(ctx: ToolContext) -> str:
    path = ctx.source.which("dmidecode") or "/usr/sbin/dmidecode"
    return (f'echo "$USER ALL=(root) NOPASSWD: {path} -t memory" | '
            "sudo tee /etc/sudoers.d/dmidecode && sudo chmod 440 /etc/sudoers.d/dmidecode "
            "&& sudo visudo -c")


def inspect_ram_slots_and_capacity(ctx: ToolContext, access: AccessLog) -> dict[str, Any]:
    tool = "inspect_ram_slots_and_capacity"
    virt = ctx.virtualization(access)

    # What is readable without root ------------------------------------- #
    no_root: dict[str, Any] = {}
    meminfo = ctx.read_full("/proc/meminfo", access)
    if meminfo:
        mi = P.parse_meminfo(meminfo)
        if mi:
            no_root["os_visible_ram_gib"] = mi["ram"]["total_gib"]
    system = _dmi_id_info(ctx, access)
    if system:
        no_root["system"] = system

    unknown_all = ["total_slots", "populated_slots", "empty_slots", "max_capacity_gib",
                   "module_type_ddr_generation", "module_speed", "upgrade_analysis"]

    # dmidecode ------------------------------------------------------------ #
    if ctx.source.is_root():
        res = ctx.run("dmidecode_memory_root", access)
    else:
        res = ctx.run("dmidecode_memory_sudo", access)
        if res.error_kind == "missing_binary" and ctx.source.which("dmidecode"):
            res = ctx.run("dmidecode_memory_root", access)   # no sudo installed

    parsed = P.parse_dmidecode_memory(res.stdout) if res.ok else {"arrays": [], "devices": []}
    has_tables = bool(parsed["arrays"] or parsed["devices"])

    if virt["virtualized"]:
        data: dict[str, Any] = {"readable_without_root": no_root}
        if has_tables:
            summary = P.summarize_memory_slots(parsed)
            summary.pop("upgrade_analysis", None)
            data["virtual_memory_devices_reported_by_hypervisor"] = summary
        return _envelope(
            ctx, tool, access, data, status="virtualized_environment",
            message=("I cannot read physical RAM slots because this is running inside a "
                     f"{'container' if virt['is_container'] else 'virtual machine'} "
                     f"({virt['type']}). The host's real motherboard DIMM slots and maximum "
                     "capacity are hidden from it. Run this on the bare-metal machine "
                     "to get that information."),
            unknown_fields=unknown_all)

    if res.ok and has_tables:
        summary = P.summarize_memory_slots(parsed, ctx.physical_ram_slots,
                                           (system or {}).get("chassis_type"))
        summary["readable_without_root"] = no_root
        notes = ["Serial numbers and asset tags are intentionally omitted for privacy."]
        return _envelope(ctx, tool, access, summary, notes=notes)

    if res.ok or res.error_kind == "no_smbios":
        return _envelope(
            ctx, tool, access, {"readable_without_root": no_root}, status="unavailable",
            message="The firmware exposes no SMBIOS memory tables, so slot count, installed "
                    "modules and maximum capacity cannot be determined.",
            unknown_fields=unknown_all)

    if res.error_kind == "permission_denied":
        return _envelope(
            ctx, tool, access, {"readable_without_root": no_root}, status="permission_required",
            message=("Root privileges are required to read physical DIMM slot information "
                     "(dmidecode reads SMBIOS tables that only root can access). "
                     "Non-interactive `sudo -n dmidecode -t memory` was refused."),
            unknown_fields=unknown_all,
            notes=["How to fix (pick one): (1) run the assistant with sudo: "
                   "`sudo .venv/bin/python main.py`; or (2) allow just this one read-only command "
                   f"without a password: `{_sudoers_hint(ctx)}`"])

    if res.error_kind == "missing_binary":
        return _envelope(
            ctx, tool, access, {"readable_without_root": no_root}, status="tool_missing",
            message="dmidecode is not installed, so physical slot information is unavailable.",
            unknown_fields=unknown_all,
            notes=["Install it with: sudo apt install dmidecode"])

    return _envelope(
        ctx, tool, access, {"readable_without_root": no_root}, status="error",
        message=f"dmidecode failed ({res.error_kind}): {res.stderr.strip()[:200]}",
        unknown_fields=unknown_all)


def _sysfs_block_disks(ctx: ToolContext, access: AccessLog) -> list[dict[str, Any]]:
    disks = []
    names = ctx.source.list_dir("/sys/block/")
    access.add(f"list /sys/block → {len(names)} entries")
    for name in names:
        if name.startswith(("loop", "ram")):
            continue
        base = f"/sys/block/{name}/"
        sectors = P.to_int(ctx.source.read_value(base + "size"))
        rota = P.to_bool(ctx.source.read_value(base + "queue/rotational"))
        model = ctx.source.read_value(base + "device/model")
        removable = P.to_bool(ctx.source.read_value(base + "removable"))
        target = (ctx.source.read_link(f"/sys/block/{name}") or "").lower()
        tran = None
        if name.startswith("nvme") or "/nvme/" in target:
            tran = "nvme"
        elif "/usb" in target:
            tran = "usb"
        elif "/ata" in target:
            tran = "sata"
        elif "virtio" in target:
            tran = "virtio"
        disks.append(P.normalize_disk(name, sectors * 512 if sectors is not None else None,
                                      rota, tran, model, removable=removable))
    return disks


def inspect_storage_devices(ctx: ToolContext, access: AccessLog) -> dict[str, Any]:
    tool = "inspect_storage_devices"
    data: dict[str, Any] = {}
    res = ctx.run("lsblk", access)
    if res.ok:
        try:
            data = P.parse_lsblk_json(res.stdout)
            data["method"] = "lsblk"
        except (ValueError, TypeError) as exc:
            log.debug("lsblk JSON parse failed: %s", exc)
    if not data:
        disks = _sysfs_block_disks(ctx, access)
        data = {"disks": disks, "method": "/sys/block (lsblk unavailable; no partition info)"}

    if not data.get("disks"):
        return _envelope(ctx, tool, access, data, status="unavailable",
                         message="No block devices could be enumerated.",
                         unknown_fields=["disks"])

    df = ctx.run("df", access)
    if df.ok:
        data["filesystem_usage"] = P.parse_df(df.stdout)

    physical = [d for d in data["disks"] if d["interface"] != "zram"]
    data["summary"] = {
        "disk_count": len(physical),
        "nvme_disks": [d["name"] for d in physical if d["is_nvme"]],
        "usb_or_removable_disks": [d["name"] for d in physical if d["removable_or_usb"]],
        "root_filesystem_disk": next(
            (d["name"] for d in physical
             if "/" in d.get("mountpoints", [])
             or any("/" in p.get("mountpoints", []) for p in d.get("partitions", []))), None),
    }
    notes = []
    if ctx.virtualization(access)["virtualized"]:
        notes.append("Running in a VM/container: these are VIRTUAL disks. Transport, model and "
                     "the rotational flag describe the hypervisor's virtual device, not the "
                     "physical drive in the host (e.g. a host NVMe SSD can appear as a "
                     "rotational 'Virtual Disk').")
    return _envelope(ctx, tool, access, data, notes=notes or None)


def inspect_pci_and_gpu(ctx: ToolContext, access: AccessLog) -> dict[str, Any]:
    tool = "inspect_pci_and_gpu"
    method = "lspci -vmm -k"
    res = ctx.run("lspci", access)
    devices = P.parse_lspci_vmm(res.stdout) if res.ok else []
    if not devices:
        method = "/sys/bus/pci/devices (lspci unavailable: vendor/class names only)"
        entries = []
        base = "/sys/bus/pci/devices/"
        for slot in ctx.source.list_dir(base):
            d = f"{base}{slot}/"
            link = ctx.source.read_link(d + "driver")
            entries.append({
                "slot": slot,
                "class_code": ctx.source.read_value(d + "class"),
                "vendor_id": ctx.source.read_value(d + "vendor"),
                "device_id": ctx.source.read_value(d + "device"),
                "revision": ctx.source.read_value(d + "revision"),
                "driver": os.path.basename(link) if link else None,
            })
        access.add(f"read {base}*/{{class,vendor,device,driver}} → {len(entries)} devices")
        devices = P.parse_pci_sysfs(entries)

    if not devices:
        return _envelope(ctx, tool, access, {}, status="unavailable",
                         message="No PCI devices could be enumerated.",
                         unknown_fields=["gpus", "network_controllers", "storage_controllers"])

    data = P.classify_pci(devices)
    other = data.pop("other_devices")
    data["other_devices_count"] = len(other)
    data["other_device_classes"] = sorted({o["class"] for o in other if o.get("class")})[:15]
    data["method"] = method

    notes = []
    if any("nvidia" in (g.get("vendor") or "").lower() for g in data["gpus"]):
        smi = ctx.run("nvidia_smi", access)
        if smi.ok:
            data["nvidia_smi"] = P.parse_nvidia_smi(smi.stdout)
        else:
            notes.append("NVIDIA GPU present but nvidia-smi is unavailable or failed, so VRAM "
                         "size is unknown (it is installed with the proprietary driver).")
    for card in ctx.source.list_dir("/sys/class/drm/"):
        if re.fullmatch(r"card\d+", card):
            vram = P.to_int(ctx.source.read_value(f"/sys/class/drm/{card}/device/mem_info_vram_total"))
            if vram:
                data.setdefault("amdgpu_vram", []).append({"card": card, "vram_gib": P.gib(vram)})
    if any(g["kind"] == "integrated" for g in data["gpus"]):
        notes.append("Integrated GPUs have no dedicated VRAM; they share system RAM.")
    if len(data["gpus"]) > 1:
        notes.append("Multiple GPUs: typical of a hybrid-graphics laptop (iGPU for the "
                     "desktop, dGPU on demand).")
    if ctx.virtualization(access)["virtualized"]:
        notes.append("Running in a VM/container: the GPU(s) listed are what the hypervisor "
                     "exposes, not necessarily the host's physical GPU.")
    if not data["gpus"]:
        notes.append("No display controller was found on the PCI bus.")
    return _envelope(ctx, tool, access, data, notes=notes or None)


def inspect_battery_and_power(ctx: ToolContext, access: AccessLog) -> dict[str, Any]:
    tool = "inspect_battery_and_power"
    base = "/sys/class/power_supply/"
    supplies = ctx.source.list_dir(base)
    access.add(f"list {base} → {len(supplies)} entries")
    batteries, adapters = [], []
    battery_attrs = ("status", "capacity", "energy_full", "energy_full_design", "energy_now",
                     "power_now", "charge_full", "charge_full_design", "charge_now",
                     "current_now", "voltage_min_design", "cycle_count", "technology",
                     "manufacturer", "model_name", "present")
    for name in supplies:
        stype = ctx.source.read_value(f"{base}{name}/type")
        if stype == "Battery":
            if (ctx.source.read_value(f"{base}{name}/scope") or "").lower() == "device":
                continue  # wireless mouse/keyboard batteries
            attrs = {a: ctx.source.read_value(f"{base}{name}/{a}") for a in battery_attrs}
            batteries.append(P.parse_battery(name, attrs))
        elif stype in ("Mains", "USB", "USB_C", "USB_PD"):
            adapters.append({"name": name, "type": stype,
                             "online": P.to_bool(ctx.source.read_value(f"{base}{name}/online"))})
    if supplies:
        access.add(f"read {base}*/{{type,status,capacity,energy_*,…}} → ok")

    profile = ctx.read("/sys/firmware/acpi/platform_profile", access)
    choices = ctx.source.read_value("/sys/firmware/acpi/platform_profile_choices")
    gov = ctx.source.read_value("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    epp = ctx.source.read_value(
        "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference")

    zones = []
    for z in ctx.source.list_dir("/sys/class/thermal/"):
        if z.startswith("thermal_zone"):
            zones.append((z, ctx.source.read_value(f"/sys/class/thermal/{z}/type"),
                          ctx.source.read_value(f"/sys/class/thermal/{z}/temp")))
    if zones:
        access.add("read /sys/class/thermal/thermal_zone*/{type,temp} → ok")

    data: dict[str, Any] = {
        "has_battery": bool(batteries),
        "batteries": batteries,
        "ac_adapters": adapters,
        "on_ac_power": any(a["online"] for a in adapters) if adapters else None,
        "platform_power_profile": profile,
        "platform_power_profile_choices": choices.split() if choices else None,
        "cpu_frequency_governor": gov,
        "cpu_energy_performance_preference": epp,
        "thermal_sensors": P.parse_thermal_zones(zones),
    }
    data = {k: v for k, v in data.items() if v not in (None, [])} | {
        "has_battery": bool(batteries)}
    notes = []
    if not batteries:
        notes.append("No system battery was found: this is a desktop, a VM/WSL, or the "
                     "battery is not exposed to Linux.")
    if batteries:
        notes.append("health_percent = full_charge_capacity / design_capacity. "
                     "estimated_time_remaining is a snapshot based on the current power draw.")
    return _envelope(ctx, tool, access, data, notes=notes or None)


def inspect_os_and_kernel(ctx: ToolContext, access: AccessLog) -> dict[str, Any]:
    tool = "inspect_os_and_kernel"
    osr = ctx.read_full("/etc/os-release", access) or ctx.read_full("/usr/lib/os-release", access)
    kernel = ctx.read("/proc/sys/kernel/osrelease", access) or P.parse_kernel_version(
        ctx.read("/proc/version", access))
    arch_res = ctx.run("uname_machine", access)
    uptime = P.parse_uptime(ctx.read("/proc/uptime", access))
    virt = ctx.virtualization(access)
    data: dict[str, Any] = {
        "distribution": P.parse_os_release(osr) if osr else None,
        "kernel_release": kernel,
        "architecture": arch_res.stdout.strip() if arch_res.ok else None,
        "boot_mode": "UEFI" if ctx.source.exists("/sys/firmware/efi") else "Legacy BIOS (or unknown)",
        "uptime": uptime["uptime"],
        "virtualization": virt,
        "running_as_root": ctx.source.is_root(),
        "system": _dmi_id_info(ctx, access) or None,
    }
    if data["distribution"]:
        data["distribution"] = {k: v for k, v in data["distribution"].items() if v}
    unknown = [k for k in ("distribution", "kernel_release", "architecture") if not data.get(k)]
    data = {k: v for k, v in data.items() if v is not None}
    return _envelope(ctx, tool, access, data, unknown_fields=unknown or None)


# --------------------------------------------------------------------------- #
# Registry / schemas / dispatcher
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    func: Callable[[ToolContext, AccessLog], dict[str, Any]]
    cacheable: bool = False


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "inspect_cpu",
        "CPU details: model name, vendor, architecture, physical core count, logical "
        "CPU/thread count, hybrid core layout, max/min clock speed (MHz), cache sizes, "
        "notable instruction-set flags, frequency governor.",
        inspect_cpu, cacheable=False),
    ToolSpec(
        "inspect_memory_usage",
        "Current RAM and swap usage right now: total (OS-visible), used, free, available, "
        "buffers/cache, swap. Use for 'how much RAM am I using / do I have free'.",
        inspect_memory_usage),
    ToolSpec(
        "inspect_ram_slots_and_capacity",
        "Physical memory hardware: number of RAM/DIMM slots, which are populated or empty, "
        "each module's size, DDR generation (DDR4/DDR5), speed and form factor, the "
        "motherboard's maximum supported RAM, soldered memory, and a precomputed upgrade "
        "analysis. Use for 'can I add/upgrade RAM', 'how much RAM can I install', "
        "'what type of RAM do I have', 'free slots'.",
        inspect_ram_slots_and_capacity, cacheable=False),
    ToolSpec(
        "inspect_storage_devices",
        "Disks and storage: each drive's model, size, interface (NVMe / SATA / USB), "
        "SSD vs HDD (rotational), partitions, mount points, and filesystem space used/free. "
        "Use for 'is my SSD NVMe', 'how much disk space is left'.",
        inspect_storage_devices),
    ToolSpec(
        "inspect_pci_and_gpu",
        "PCI devices: GPUs (integrated/discrete, vendor, model, driver, VRAM for NVIDIA/AMD), "
        "network/Wi-Fi/Ethernet controllers, NVMe/SATA storage controllers, audio devices.",
        inspect_pci_and_gpu),
    ToolSpec(
        "inspect_battery_and_power",
        "Battery and power: charge %, charging/discharging status, battery health (wear) "
        "vs design capacity, cycle count, power draw, estimated time remaining, AC adapter "
        "state, power profile, CPU governor, and temperature sensors.",
        inspect_battery_and_power),
    ToolSpec(
        "inspect_os_and_kernel",
        "Operating system and platform: Linux distribution and version, kernel release, "
        "CPU architecture, UEFI vs legacy boot, uptime, virtualization/VM/WSL detection, "
        "and the computer's manufacturer/model/BIOS.",
        inspect_os_and_kernel, cacheable=False),
)


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI-compatible ``tools=[...]`` definitions (Groq and Ollama accept these)."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
        for spec in TOOL_SPECS
    ]


def _shrink(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep tool results within a token budget (free tiers / small models)."""
    text = json.dumps(payload, separators=(",", ":"), default=str)
    if len(text) <= MAX_PAYLOAD_CHARS:
        return payload
    data = payload.get("data", {})
    for key in ("filesystem_usage", "thermal_sensors", "network_controllers", "audio_devices",
                "storage_controllers", "partitions"):
        if isinstance(data.get(key), list) and len(data[key]) > 5:
            data[key] = data[key][:5] + [{"truncated": len(data[key]) - 5}]
    payload["truncated"] = True
    return payload


class ToolRegistry:
    """Maps tool names → inspectors, with per-session caching of static data."""

    def __init__(self, source: SystemSource, physical_ram_slots: int | None = None) -> None:
        self.ctx = ToolContext(source, physical_ram_slots)
        self._specs = {s.name: s for s in TOOL_SPECS}
        self._cache: dict[str, ToolOutcome] = {}

    @property
    def names(self) -> list[str]:
        return list(self._specs)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return tool_schemas()

    def describe(self) -> list[tuple[str, str]]:
        return [(s.name, s.description) for s in TOOL_SPECS]

    def clear_cache(self) -> None:
        self._cache.clear()

    def dispatch(self, name: str, arguments: str | dict | None = None) -> ToolOutcome:
        spec = self._specs.get(name)
        if spec is None:
            return ToolOutcome(name=name, payload={
                "tool": name, "status": "error",
                "message": f"Unknown tool '{name}'. Available tools: {', '.join(self.names)}."})

        # These tools take no parameters. Arguments are parsed only to log
        # anomalies – they can never influence what is executed.
        if arguments not in (None, "", "{}", {}):
            try:
                args = json.loads(arguments) if isinstance(arguments, str) else arguments
                if args:
                    log.debug("Ignoring unexpected arguments for %s: %r", name, args)
            except (TypeError, ValueError):
                log.debug("Ignoring malformed arguments for %s: %r", name, arguments)

        if spec.cacheable and name in self._cache:
            cached = self._cache[name]
            return ToolOutcome(name=name, payload=cached.payload,
                               accessed=["(cached result from earlier in this session)"],
                               cached=True)

        access = AccessLog()
        started = time.monotonic()
        try:
            payload = spec.func(self.ctx, access)
        except Exception as exc:  # never let one tool crash the chat
            log.exception("Tool %s crashed", name)
            payload = {"tool": name, "status": "error",
                       "message": f"Internal error while inspecting: {type(exc).__name__}: {exc}"}
        outcome = ToolOutcome(name=name, payload=_shrink(payload), accessed=access.entries,
                              duration_ms=int((time.monotonic() - started) * 1000))
        if spec.cacheable and outcome.status != "error":
            self._cache[name] = outcome
        return outcome
