"""Pure parsers: raw tool text  →  clean, structured dictionaries.

Every function here is side-effect free (no I/O), which makes them trivially
unit-testable against fixtures. They are tolerant of format differences
between util-linux / procps / dmidecode versions and never raise on odd
input – unknown values become ``None`` instead.

Privacy: identifiers (DIMM serial numbers, asset tags, usernames inside
mount paths) are dropped or redacted here, *before* anything reaches an LLM.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

KIB = 1024
MIB = 1024 ** 2
GIB = 1024 ** 3

# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #

_BIN_UNITS = {
    "": 1, "B": 1,
    "K": KIB, "KB": KIB, "KIB": KIB,
    "M": MIB, "MB": MIB, "MIB": MIB,
    "G": GIB, "GB": GIB, "GIB": GIB,
    "T": 1024 ** 4, "TB": 1024 ** 4, "TIB": 1024 ** 4,
    "P": 1024 ** 5, "PB": 1024 ** 5, "PIB": 1024 ** 5,
}
_SIZE_RE = re.compile(r"^\s*([\d.]+)\s*([KMGTP]?i?B?)\b", re.IGNORECASE)
_UNKNOWN = {"", "unknown", "not specified", "not provided", "none", "n/a", "[n/a]",
            "to be filled by o.e.m.", "default string", "-"}


def clean(value: Any) -> str | None:
    """Normalise placeholder strings ("Unknown", "Not Specified", ...) to None."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in _UNKNOWN else text


def to_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return None


def to_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def parse_mem_size_to_bytes(text: str | None) -> int | None:
    """Parse memory-style sizes (binary units): '16 GB', '544 KiB', '1280K'."""
    if text is None:
        return None
    m = _SIZE_RE.match(str(text))
    if not m:
        return None
    number = to_float(m.group(1))
    unit = m.group(2).upper()
    if number is None or unit not in _BIN_UNITS:
        return None
    return int(number * _BIN_UNITS[unit])


def gib(n_bytes: int | None, digits: int = 2) -> float | None:
    return None if n_bytes is None else round(n_bytes / GIB, digits)


def gb_decimal(n_bytes: int | None, digits: int = 1) -> float | None:
    return None if n_bytes is None else round(n_bytes / 1e9, digits)


def human_bytes(n_bytes: int | None) -> str | None:
    if n_bytes is None:
        return None
    value = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024 or unit == "PiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return None  # pragma: no cover


def redact_path(path: str | None) -> str | None:
    """Hide the username in /home/<user>, /media/<user>, /run/media/<user>."""
    if not path:
        return path
    return re.sub(r"^/(home|media|run/media)/[^/]+", lambda m: f"/{m.group(1)}/<user>", path)


def _kv_lines(text: str, sep: str = ":") -> Iterable[tuple[str, str]]:
    for line in text.splitlines():
        if sep not in line:
            continue
        key, _, value = line.partition(sep)
        yield key.strip(), value.strip()


# --------------------------------------------------------------------------- #
# CPU
# --------------------------------------------------------------------------- #

FLAGS_OF_INTEREST = (
    "avx", "avx2", "avx512f", "avx_vnni", "aes", "sha_ni", "vmx", "svm", "hypervisor",
    "ht", "lm", "nx",
)


def _interesting_flags(flags: str | None) -> list[str]:
    present = set((flags or "").split())
    return [f for f in FLAGS_OF_INTEREST if f in present]


def _parse_cache(value: str) -> dict[str, Any]:
    # "11.5 MiB (8 instances)"  |  "1280K"  |  "48 KiB"
    instances = None
    m = re.search(r"\((\d+)\s+instances?\)", value)
    if m:
        instances = int(m.group(1))
    size_text = value.split("(")[0].strip()
    out: dict[str, Any] = {"total": size_text}
    if instances:
        out["instances"] = instances
    return out


def parse_lscpu(text: str) -> dict[str, Any]:
    """Parse ``lscpu`` (flat legacy *or* indented tree output, ``LC_ALL=C``)."""
    fields: dict[str, list[str]] = {}
    for key, value in _kv_lines(text):
        fields.setdefault(key, []).append(value)

    def first(*keys: str) -> str | None:
        for k in keys:
            for v in fields.get(k, []):
                if clean(v):
                    return v
        return None

    model_names = list(dict.fromkeys(v for v in fields.get("Model name", []) if clean(v)))
    sockets = to_int(first("Socket(s)"))
    cores_per_socket = to_int(first("Core(s) per socket"))
    clusters = to_int(first("Cluster(s)"))
    cores_per_cluster = to_int(first("Core(s) per cluster"))
    physical = None
    if sockets and cores_per_socket:
        physical = sockets * cores_per_socket
    elif clusters and cores_per_cluster:
        physical = clusters * cores_per_cluster

    caches: dict[str, Any] = {}
    for level in ("L1d", "L1i", "L2", "L3", "L4"):
        raw = first(f"{level} cache", level)
        if raw:
            caches[level] = _parse_cache(raw)

    return {
        "model_name": model_names[0] if model_names else None,
        "model_names_all": model_names if len(model_names) > 1 else None,
        "vendor": clean(first("Vendor ID")),
        "architecture": clean(first("Architecture")),
        "op_modes": clean(first("CPU op-mode(s)")),
        "address_sizes": clean(first("Address sizes")),
        "logical_cpus": to_int(first("CPU(s)")),
        "sockets": sockets,
        "cores_per_socket": cores_per_socket,
        "threads_per_core_reported": to_int(first("Thread(s) per core")),
        "physical_cores": physical,
        "max_mhz": to_float(first("CPU max MHz")),
        "min_mhz": to_float(first("CPU min MHz")),
        "virtualization_support": clean(first("Virtualization")),
        "hypervisor_vendor": clean(first("Hypervisor vendor")),
        "numa_nodes": to_int(first("NUMA node(s)")),
        "caches": caches,
        "notable_flags": _interesting_flags(first("Flags")),
    }


def parse_lscpu_topology(text: str) -> dict[str, Any]:
    """Parse ``lscpu -p=CPU,CORE,SOCKET,NODE`` for exact core/thread counts.

    Detects hybrid designs (e.g. Intel P-cores with SMT + E-cores without),
    where "Thread(s) per core" from plain ``lscpu`` is misleading.
    """
    threads_per_core: dict[tuple[str, str], int] = {}
    sockets: set[str] = set()
    cpus = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        cpus += 1
        core, socket = parts[1], parts[2]
        sockets.add(socket)
        threads_per_core[(socket, core)] = threads_per_core.get((socket, core), 0) + 1
    if not cpus:
        return {}
    with_smt = sum(1 for n in threads_per_core.values() if n > 1)
    without_smt = sum(1 for n in threads_per_core.values() if n == 1)
    return {
        "logical_cpus": cpus,
        "physical_cores": len(threads_per_core),
        "sockets": len(sockets),
        "cores_with_smt": with_smt,
        "cores_without_smt": without_smt,
        "hybrid_core_layout": with_smt > 0 and without_smt > 0,
    }


def parse_proc_cpuinfo(text: str) -> dict[str, Any]:
    """Fallback CPU parser for ``/proc/cpuinfo`` (x86 and ARM)."""
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    processors: list[dict[str, str]] = []
    header: dict[str, str] = {}
    for block in blocks:
        entry = {k.lower(): v for k, v in _kv_lines(block)}
        if "processor" in entry and to_int(entry["processor"]) is not None:
            processors.append(entry)
        else:
            header.update(entry)  # ARM puts "Hardware", "Serial" etc. at the end

    if not processors:
        return {}
    first = processors[0]
    core_keys = {(p.get("physical id", "0"), p.get("core id")) for p in processors
                 if p.get("core id") is not None}
    socket_ids = {p.get("physical id") for p in processors if p.get("physical id") is not None}
    mhz = [to_float(p.get("cpu mhz")) for p in processors if to_float(p.get("cpu mhz"))]
    model = (clean(first.get("model name")) or clean(header.get("model name"))
             or clean(header.get("hardware")) or clean(first.get("cpu part")))
    flags = first.get("flags") or first.get("features")
    return {
        "model_name": model,
        "vendor": clean(first.get("vendor_id")) or clean(first.get("cpu implementer")),
        "logical_cpus": len(processors),
        "physical_cores": len(core_keys) or None,
        "sockets": len(socket_ids) or None,
        "cache_size_reported": clean(first.get("cache size")),
        "current_mhz_min": round(min(mhz), 1) if mhz else None,
        "current_mhz_max": round(max(mhz), 1) if mhz else None,
        "notable_flags": _interesting_flags(flags),
    }


# --------------------------------------------------------------------------- #
# Memory usage
# --------------------------------------------------------------------------- #

def _usage_block(total: int | None, used: int | None, free: int | None,
                 available: int | None, shared: int | None = None,
                 buff_cache: int | None = None) -> dict[str, Any]:
    pct = round(100 * used / total, 1) if total and used is not None else None
    block = {
        "total_gib": gib(total), "used_gib": gib(used), "free_gib": gib(free),
        "available_gib": gib(available), "shared_gib": gib(shared),
        "buff_cache_gib": gib(buff_cache), "used_percent": pct,
    }
    return {k: v for k, v in block.items() if v is not None}


def parse_free(text: str) -> dict[str, Any]:
    """Parse ``free -b`` (handles both modern and legacy procps layouts)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {}
    headers = lines[0].split()
    rows: dict[str, dict[str, int | None]] = {}
    for line in lines[1:]:
        label, _, rest = line.partition(":")
        values = rest.split()
        rows[label.strip().lower()] = {h: to_int(v) for h, v in zip(headers, values)}

    mem = rows.get("mem", {})
    swap = rows.get("swap", {})
    if not mem:
        return {}
    buff_cache = mem.get("buff/cache")
    if buff_cache is None and mem.get("buffers") is not None:
        buff_cache = (mem.get("buffers") or 0) + (mem.get("cached") or 0)
    return {
        "ram": _usage_block(mem.get("total"), mem.get("used"), mem.get("free"),
                            mem.get("available"), mem.get("shared"), buff_cache),
        "swap": _usage_block(swap.get("total"), swap.get("used"), swap.get("free"), None)
        if swap else None,
    }


def parse_meminfo(text: str) -> dict[str, Any]:
    """Fallback: ``/proc/meminfo`` → same shape as :func:`parse_free`."""
    kv: dict[str, int] = {}
    for key, value in _kv_lines(text):
        parts = value.split()
        if not parts:
            continue
        number = to_int(parts[0])
        if number is None:
            continue
        kv[key] = number * 1024 if len(parts) > 1 and parts[1].lower() == "kb" else number
    if "MemTotal" not in kv:
        return {}
    total, free = kv["MemTotal"], kv.get("MemFree")
    buff_cache = kv.get("Buffers", 0) + kv.get("Cached", 0) + kv.get("SReclaimable", 0)
    used = total - (free or 0) - buff_cache if free is not None else None
    swap_total, swap_free = kv.get("SwapTotal"), kv.get("SwapFree")
    swap_used = swap_total - swap_free if swap_total is not None and swap_free is not None else None
    return {
        "ram": _usage_block(total, used, free, kv.get("MemAvailable"), kv.get("Shmem"), buff_cache),
        "swap": _usage_block(swap_total, swap_used, swap_free, None),
    }


# --------------------------------------------------------------------------- #
# Physical memory (dmidecode)
# --------------------------------------------------------------------------- #

_NOT_INSTALLED = ("no module installed", "not installed", "0 b", "0 mb", "0 gb")
_SOLDERED_FORM_FACTORS = {"row of chips", "chip"}


def _dmi_blocks(text: str) -> list[tuple[int, dict[str, str]]]:
    blocks: list[tuple[int, dict[str, str]]] = []
    current: dict[str, str] | None = None
    dmi_type = -1
    for raw in text.splitlines():
        m = re.match(r"^Handle\s+(0x[0-9A-Fa-f]+),\s*DMI type\s+(\d+)", raw)
        if m:
            current = {"handle": m.group(1)}
            dmi_type = int(m.group(2))
            blocks.append((dmi_type, current))
            continue
        if current is None:
            continue
        if raw.startswith("\t") and not raw.startswith("\t\t") and ":" in raw:
            key, _, value = raw.strip().partition(":")
            current[key.strip()] = value.strip()
    return blocks


def _speed(value: str | None) -> int | None:
    m = re.match(r"^\s*(\d+)\s*(MT/s|MHz)", value or "")
    return int(m.group(1)) if m else None


def parse_dmidecode_memory(text: str) -> dict[str, Any]:
    """Parse ``dmidecode -t memory`` (SMBIOS types 16 and 17).

    Serial numbers and asset tags are intentionally NOT returned.
    """
    arrays: list[dict[str, Any]] = []
    devices: list[dict[str, Any]] = []
    for dmi_type, block in _dmi_blocks(text):
        if dmi_type == 16:
            max_cap = parse_mem_size_to_bytes(block.get("Maximum Capacity"))
            ext = parse_mem_size_to_bytes(block.get("Extended Maximum Capacity"))
            arrays.append({
                "handle": block["handle"],
                "location": clean(block.get("Location")),
                "use": clean(block.get("Use")),
                "error_correction": clean(block.get("Error Correction Type")),
                "max_capacity_bytes": ext or max_cap,
                "number_of_devices": to_int(block.get("Number Of Devices")),
            })
        elif dmi_type == 17:
            size_raw = (block.get("Size") or "").strip()
            installed = bool(size_raw) and size_raw.lower() not in _NOT_INSTALLED \
                and not size_raw.lower().startswith("unknown")
            form = clean(block.get("Form Factor"))
            mtype = clean(block.get("Type"))
            soldered = (installed and (
                (form or "").lower() in _SOLDERED_FORM_FACTORS
                or (mtype or "").upper().startswith("LPDDR")))
            devices.append({
                "array_handle": block.get("Array Handle"),
                "locator": clean(block.get("Locator")),
                "bank_locator": clean(block.get("Bank Locator")),
                "installed": installed,
                "size_bytes": parse_mem_size_to_bytes(size_raw) if installed else None,
                "size_gib": gib(parse_mem_size_to_bytes(size_raw)) if installed else None,
                "form_factor": form if installed else None,
                "type": mtype if installed else None,
                "speed_mts": _speed(block.get("Speed")) if installed else None,
                "configured_speed_mts": _speed(block.get("Configured Memory Speed")
                                               or block.get("Configured Clock Speed"))
                if installed else None,
                "manufacturer": clean(block.get("Manufacturer")) if installed else None,
                "part_number": clean(block.get("Part Number")) if installed else None,
                "rank": to_int(block.get("Rank")) if installed else None,
                "soldered_onboard": soldered,
            })
    return {"arrays": arrays, "devices": devices}


PORTABLE_CHASSIS = {"Portable", "Laptop", "Notebook", "Sub Notebook", "Convertible",
                    "Detachable", "Hand Held", "Tablet"}


def summarize_memory_slots(parsed: dict[str, Any], physical_slots: int | None = None,
                           chassis_type: str | None = None) -> dict[str, Any]:
    """Turn parsed dmidecode data into verified facts + a deterministic
    upgrade analysis, so the LLM never has to do the arithmetic itself.

    ``physical_slots`` is the user's own count of real slots (PHYSICAL_RAM_SLOTS).
    Firmware often lists "phantom" slots the board does not have – e.g. many
    laptops report 4 memory devices (2 channels x 2) while only 2 slots exist.
    """
    arrays = [a for a in parsed.get("arrays", [])
              if (a.get("use") or "System Memory").lower() == "system memory"]
    handles = {a["handle"] for a in arrays}
    devices = [d for d in parsed.get("devices", [])
               if not handles or d.get("array_handle") in handles]

    declared_slots = sum(a.get("number_of_devices") or 0 for a in arrays)
    total_slots = declared_slots or len(devices)
    modules = [d for d in devices if d["installed"]]
    empty = [d for d in devices if not d["installed"]]
    reported_slots = total_slots
    n_empty = len(empty)
    empty_locators: list[str | None] | None = [d["locator"] for d in empty]
    slot_source = "firmware (SMBIOS)"
    slot_warning = None
    if physical_slots and physical_slots >= len(modules):
        total_slots = physical_slots
        n_empty = physical_slots - len(modules)
        slot_source = "user setting (--ram-slots or PHYSICAL_RAM_SLOTS; overrides firmware)"
        if physical_slots != reported_slots:
            empty_locators = None  # firmware names don't map to the real slots
    elif chassis_type in PORTABLE_CHASSIS and total_slots > 2:
        slot_warning = (
            f"Firmware reports {total_slots} memory slots, but this is a {chassis_type} and most "
            "laptops physically have 1-2 slots. Consumer-laptop BIOSes often list 'phantom' "
            "slots (what the memory controller supports, not what the board has), so the slot "
            "count and the per-slot estimate below may be wrong. Check the real number of slots "
            "(service manual or under the RAM cover) and set PHYSICAL_RAM_SLOTS in .env.")
    installed_bytes = sum(d["size_bytes"] or 0 for d in modules)
    max_caps = [a["max_capacity_bytes"] for a in arrays if a.get("max_capacity_bytes")]
    max_bytes = sum(max_caps) if max_caps else None
    soldered = [d for d in modules if d["soldered_onboard"]]
    socketed_slots = total_slots - len(soldered)

    types = sorted({d["type"] for d in modules if d["type"]})
    forms = sorted({d["form_factor"] for d in modules if d["form_factor"]})
    speeds = sorted({d["speed_mts"] for d in modules if d["speed_mts"]})

    analysis: dict[str, Any] = {"can_compute": max_bytes is not None and total_slots > 0}
    caveats = [
        "maximum_capacity comes from SMBIOS Type 16 (firmware). It often reflects the CPU "
        "memory-controller limit and can differ from the manufacturer's validated maximum; "
        "confirm with the laptop's service manual before buying.",
    ]
    if analysis["can_compute"]:
        headroom = max(0, max_bytes - installed_bytes)
        per_slot_est = max_bytes // socketed_slots if socketed_slots > 0 else 0
        addable_without_replacing = min(headroom, n_empty * per_slot_est)
        analysis.update({
            "headroom_gib": gib(headroom),
            "empty_slots": n_empty,
            "can_add_ram": headroom > 0 and (n_empty > 0 or socketed_slots > 0),
            "estimated_max_per_slot_gib": gib(per_slot_est),
            "max_addable_without_replacing_gib": gib(addable_without_replacing),
            "total_after_filling_empty_slots_gib": gib(installed_bytes + addable_without_replacing),
            "to_reach_maximum": (
                f"{socketed_slots} x {gib(per_slot_est):g} GiB modules "
                f"(replacing existing socketed modules) = {gib(max_bytes):g} GiB"
                if socketed_slots > 0 and soldered == [] else
                "Some memory is soldered; only socketed slots can be changed."
            ),
            "match_existing": {"type": types, "form_factor": forms, "speed_mts": speeds},
        })
        caveats.append("estimated_max_per_slot = max_capacity / socketed slots. This is a "
                       "common heuristic, NOT a value reported by firmware.")
    if slot_warning:
        caveats.insert(0, slot_warning)
    if soldered:
        caveats.append(f"{len(soldered)} module(s) are soldered to the motherboard and "
                       "cannot be removed or replaced.")
    analysis["caveats"] = caveats

    out: dict[str, Any] = {
        "total_slots": total_slots,
        "slot_count_source": slot_source,
        "populated_slots": len(modules),
        "empty_slots": n_empty,
    }
    if reported_slots != total_slots:
        out["slots_reported_by_firmware"] = reported_slots
    if slot_warning:
        out["slot_count_reliable"] = False
        out["slot_count_warning"] = slot_warning
    if empty_locators is not None:
        out["empty_slot_locators"] = empty_locators
    return out | {
        "installed_gib": gib(installed_bytes),
        "max_capacity_gib": gib(max_bytes),
        "error_correction": sorted({a["error_correction"] for a in arrays
                                    if a.get("error_correction")}),
        "modules": [{k: v for k, v in d.items()
                     if k not in ("array_handle", "installed", "size_bytes") and v is not None}
                    for d in modules],
        "upgrade_analysis": analysis,
    }


CHASSIS_TYPES = {
    "1": "Other", "2": "Unknown", "3": "Desktop", "4": "Low Profile Desktop",
    "5": "Pizza Box", "6": "Mini Tower", "7": "Tower", "8": "Portable", "9": "Laptop",
    "10": "Notebook", "11": "Hand Held", "12": "Docking Station", "13": "All in One",
    "14": "Sub Notebook", "15": "Space-saving", "16": "Lunch Box", "17": "Main Server Chassis",
    "23": "Rack Mount Chassis", "24": "Sealed-case PC", "30": "Tablet", "31": "Convertible",
    "32": "Detachable", "33": "IoT Gateway", "34": "Embedded PC", "35": "Mini PC",
    "36": "Stick PC",
}


def parse_dmi_id(values: dict[str, str | None]) -> dict[str, Any]:
    """Normalise the world-readable files in ``/sys/devices/virtual/dmi/id``."""
    chassis = clean(values.get("chassis_type"))
    return {
        "system_vendor": clean(values.get("sys_vendor")),
        "product_name": clean(values.get("product_name")),
        "product_version": clean(values.get("product_version")),
        "product_family": clean(values.get("product_family")),
        "board_vendor": clean(values.get("board_vendor")),
        "board_name": clean(values.get("board_name")),
        "bios_vendor": clean(values.get("bios_vendor")),
        "bios_version": clean(values.get("bios_version")),
        "bios_date": clean(values.get("bios_date")),
        "chassis_type": CHASSIS_TYPES.get(chassis or "", chassis),
    }


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #

def _infer_interface(name: str, tran: str | None) -> str:
    if tran:
        return tran.lower()
    if name.startswith("nvme"):
        return "nvme"
    if name.startswith("mmcblk"):
        return "mmc"
    if name.startswith("vd"):
        return "virtio"
    if name.startswith("xvd"):
        return "xen"
    if name.startswith("zram"):
        return "zram"
    return "unknown"


def _media_type(interface: str, rota: bool | None) -> str:
    if interface == "zram":
        return "compressed RAM swap device (zram) – not a physical disk"
    if interface == "nvme":
        return "NVMe SSD"
    if interface == "mmc":
        return "eMMC / SD flash"
    if rota is False:
        return "SSD / flash (non-rotational)"
    if rota is True:
        return "HDD (rotational)"
    return "unknown"


def _mounts(dev: dict[str, Any]) -> list[str]:
    raw = dev.get("mountpoints")
    if raw is None:
        raw = [dev.get("mountpoint")]
    return [redact_path(m) for m in raw if m]


def normalize_disk(name: str, size_bytes: int | None, rota: bool | None, tran: str | None,
                   model: str | None, partitions: list[dict[str, Any]] | None = None,
                   removable: bool | None = None) -> dict[str, Any]:
    interface = _infer_interface(name, tran)
    return {
        "name": name,
        "model": clean(model),
        "size_gb_decimal": gb_decimal(size_bytes),
        "size_gib": gib(size_bytes),
        "transport": tran,
        "interface": interface,
        "is_nvme": interface == "nvme",
        "rotational": rota,
        "media_type": _media_type(interface, rota),
        "removable_or_usb": bool(removable) or interface == "usb",
        "partitions": partitions or [],
    }


def parse_lsblk_json(text: str) -> dict[str, Any]:
    """Parse ``lsblk -J -b -o NAME,SIZE,TYPE,MOUNTPOINT,ROTA,TRAN,MODEL``.

    Works with old util-linux (strings, "0"/"1") and new (numbers, booleans).
    """
    data = json.loads(text)
    disks: list[dict[str, Any]] = []
    loops = 0
    other: list[dict[str, Any]] = []
    for dev in data.get("blockdevices", []):
        dtype = (dev.get("type") or "").lower()
        name = dev.get("name") or "?"
        if dtype == "loop":
            loops += 1
            continue
        if dtype not in {"disk", "rom"}:
            other.append({"name": name, "type": dtype})
            continue
        parts = []
        for child in dev.get("children") or []:
            parts.append({
                "name": child.get("name"),
                "type": child.get("type"),
                "size_gib": gib(to_int(child.get("size"))),
                "mountpoints": _mounts(child),
            })
        disk = normalize_disk(name, to_int(dev.get("size")), to_bool(dev.get("rota")),
                              clean(dev.get("tran")), dev.get("model"), parts)
        disk["mountpoints"] = _mounts(dev)
        disk["device_type"] = dtype
        disks.append(disk)
    return {"disks": disks, "loop_devices_skipped": loops, "other_devices": other}


def parse_df(text: str) -> list[dict[str, Any]]:
    """Parse ``df -B1 --output=source,fstype,size,used,avail,target``."""
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        source, fstype, size, used, avail, target = parts
        if source.startswith("/dev/loop"):
            continue
        size_i, used_i, avail_i = to_int(size), to_int(used), to_int(avail)
        pct = None
        if used_i is not None and avail_i is not None and (used_i + avail_i) > 0:
            pct = round(100 * used_i / (used_i + avail_i), 1)
        rows.append({
            "source": source,
            "fstype": fstype,
            "mountpoint": redact_path(target.strip()),
            "size_gib": gib(size_i),
            "used_gib": gib(used_i),
            "available_gib": gib(avail_i),
            "used_percent": pct,
        })
    return rows


# --------------------------------------------------------------------------- #
# PCI / GPU
# --------------------------------------------------------------------------- #

def parse_lspci_vmm(text: str) -> list[dict[str, Any]]:
    """Parse ``lspci -vmm -k`` (blank-line separated ``Key:\\tValue`` records)."""
    devices: list[dict[str, Any]] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        rec: dict[str, Any] = {"modules": []}
        for line in block.splitlines():
            key, sep, value = line.partition(":")
            if not sep:
                continue
            key, value = key.strip(), value.strip()
            if key == "Module":
                rec["modules"].append(value)
            elif key == "Slot":
                rec["slot"] = value
            else:
                rec[key.lower()] = value
        if "slot" in rec:
            devices.append({
                "slot": rec.get("slot"),
                "class": rec.get("class"),
                "vendor": rec.get("vendor"),
                "device": rec.get("device"),
                "subsystem_vendor": clean(rec.get("svendor")),
                "subsystem_device": clean(rec.get("sdevice")),
                "revision": rec.get("rev"),
                "driver": rec.get("driver"),
                "kernel_modules": rec["modules"],
            })
    return devices


PCI_CLASS_NAMES = {
    "0100": "SCSI storage controller", "0101": "IDE interface", "0104": "RAID bus controller",
    "0106": "SATA controller", "0107": "Serial Attached SCSI controller",
    "0108": "Non-Volatile memory controller", "0180": "Mass storage controller",
    "0200": "Ethernet controller", "0280": "Network controller",
    "0300": "VGA compatible controller", "0302": "3D controller", "0380": "Display controller",
    "0401": "Multimedia audio controller", "0403": "Audio device",
    "0480": "Multimedia controller", "0500": "RAM memory", "0580": "Memory controller",
    "0600": "Host bridge", "0601": "ISA bridge", "0604": "PCI bridge",
    "0700": "Serial controller", "0780": "Communication controller",
    "0880": "System peripheral", "0c03": "USB controller", "0c05": "SMBus",
    "0c80": "Serial bus controller", "0d11": "Bluetooth", "1180": "Signal processing controller",
}

PCI_VENDOR_NAMES = {
    "0x8086": "Intel Corporation", "0x8087": "Intel Corporation",
    "0x10de": "NVIDIA Corporation", "0x1002": "Advanced Micro Devices, Inc. [AMD/ATI]",
    "0x1022": "Advanced Micro Devices, Inc. [AMD]", "0x144d": "Samsung Electronics Co Ltd",
    "0x10ec": "Realtek Semiconductor Co., Ltd.", "0x14e4": "Broadcom Inc.",
    "0x168c": "Qualcomm Atheros", "0x17cb": "Qualcomm Technologies, Inc",
    "0x15b7": "Sandisk Corp / Western Digital", "0x1e0f": "KIOXIA Corporation",
    "0x1c5c": "SK hynix", "0x1987": "Phison Electronics Corporation",
    "0x126f": "Silicon Motion, Inc.", "0x1cc1": "ADATA Technology Co., Ltd.",
    "0x2646": "Kingston Technology Company, Inc.", "0x1b4b": "Marvell Technology Group Ltd.",
    "0x1414": "Microsoft Corporation", "0x1af4": "Red Hat, Inc. (virtio)",
    "0x15ad": "VMware", "0x80ee": "InnoTek (VirtualBox)", "0x1234": "QEMU",
    "0x1b36": "Red Hat, Inc. (QEMU)", "0x14c3": "MEDIATEK Corp.", "0x1217": "O2 Micro, Inc.",
}


def parse_pci_sysfs(entries: list[dict[str, str | None]]) -> list[dict[str, Any]]:
    """Fallback when ``lspci`` is missing: entries gathered from
    ``/sys/bus/pci/devices/*/{class,vendor,device}`` + ``driver`` symlink."""
    devices = []
    for e in entries:
        class_code = (e.get("class_code") or "").lower().replace("0x", "")
        vendor_id = (e.get("vendor_id") or "").lower()
        device_id = (e.get("device_id") or "").lower()
        devices.append({
            "slot": e.get("slot"),
            "class": PCI_CLASS_NAMES.get(class_code[:4], f"PCI class 0x{class_code[:4]}"),
            "vendor": PCI_VENDOR_NAMES.get(vendor_id, f"vendor {vendor_id}"),
            "device": (f"device {vendor_id[2:]}:{device_id[2:]} "
                       "(model name unavailable without lspci)"),
            "subsystem_vendor": None,
            "subsystem_device": None,
            "revision": e.get("revision"),
            "driver": e.get("driver"),
            "kernel_modules": [],
        })
    return devices


_GPU_CLASSES = {"vga compatible controller", "3d controller", "display controller"}
_NET_CLASSES = {"ethernet controller", "network controller"}
_STORAGE_CLASSES = {"non-volatile memory controller", "sata controller", "raid bus controller",
                    "scsi storage controller", "ide interface", "mass storage controller",
                    "serial attached scsi controller"}
_VIRTUAL_GPU_VENDORS = ("microsoft", "vmware", "red hat", "innotek", "virtualbox", "qemu",
                        "cirrus", "bochs")


def _gpu_kind(dev: dict[str, Any]) -> str:
    vendor = (dev.get("vendor") or "").lower()
    name = (dev.get("device") or "").lower()
    if any(v in vendor for v in _VIRTUAL_GPU_VENDORS) or "basic render" in name:
        return "virtual (provided by hypervisor)"
    if "nvidia" in vendor:
        return "discrete"
    if "intel" in vendor:
        return "discrete" if "arc" in name and "graphics" not in name else "integrated"
    if "amd" in vendor or "ati" in vendor:
        if any(k in name for k in ("radeon rx", "navi 1", "navi 2", "navi 3", "radeon pro")):
            return "discrete (likely)"
        return "integrated or discrete (cannot tell from PCI data)"
    return "unknown"


def classify_pci(devices: list[dict[str, Any]]) -> dict[str, Any]:
    gpus, net, storage, audio, other = [], [], [], [], []
    for d in devices:
        cls = (d.get("class") or "").lower()
        if cls in _GPU_CLASSES:
            g = dict(d)
            g["kind"] = _gpu_kind(d)
            drv = d.get("driver")
            g["driver_note"] = (
                "no kernel driver bound" if not drv else
                "proprietary NVIDIA driver" if drv == "nvidia" else
                "open-source nouveau driver" if drv == "nouveau" else f"kernel driver '{drv}'")
            gpus.append(g)
        elif cls in _NET_CLASSES:
            net.append(d)
        elif cls in _STORAGE_CLASSES:
            storage.append(d)
        elif "audio" in cls:
            audio.append(d)
        else:
            other.append({"slot": d.get("slot"), "class": d.get("class"),
                          "vendor": d.get("vendor"), "device": d.get("device")})
    def compact(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: d.get(k) for k in ("slot", "class", "vendor", "device", "driver")
                 if d.get(k)} for d in items]
    for g in gpus:
        for k in ("revision", "subsystem_device"):
            g.pop(k, None)
        if not g.get("subsystem_vendor"):
            g.pop("subsystem_vendor", None)
    net, storage, audio = compact(net), compact(storage), compact(audio)
    return {"gpus": gpus, "network_controllers": net, "storage_controllers": storage,
            "audio_devices": audio, "other_devices": other}


def parse_nvidia_smi(text: str) -> list[dict[str, Any]]:
    """Parse ``nvidia-smi --query-gpu=name,memory.total,driver_version,
    temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits``."""
    gpus = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        parts += [None] * (6 - len(parts))
        vram_mib = to_int(parts[1]) if clean(parts[1]) else None
        gpus.append({
            "name": parts[0],
            "vram_mib": vram_mib,
            "vram_gib": round(vram_mib / 1024, 2) if vram_mib else None,
            "driver_version": clean(parts[2]),
            "temperature_c": to_float(parts[3]) if clean(parts[3]) else None,
            "power_draw_w": to_float(parts[4]) if clean(parts[4]) else None,
            "utilization_percent": to_float(parts[5]) if clean(parts[5]) else None,
        })
    return gpus


# --------------------------------------------------------------------------- #
# Power / battery / thermal
# --------------------------------------------------------------------------- #

def _hours_to_text(hours: float | None) -> str | None:
    if hours is None or hours < 0 or hours > 100:
        return None
    h = int(hours)
    m = int(round((hours - h) * 60))
    if m == 60:
        h, m = h + 1, 0
    return f"{h}h {m:02d}m"


def parse_battery(name: str, attrs: dict[str, str | None]) -> dict[str, Any]:
    """Normalise one ``/sys/class/power_supply/BAT*`` directory.

    Handles both ``energy_*`` (µWh) and ``charge_*`` (µAh) reporting styles.
    """
    status = clean(attrs.get("status"))
    v_design = to_int(attrs.get("voltage_min_design"))
    energy_full = to_int(attrs.get("energy_full"))
    energy_design = to_int(attrs.get("energy_full_design"))
    energy_now = to_int(attrs.get("energy_now"))
    power_now = to_int(attrs.get("power_now"))
    if energy_full is None and attrs.get("charge_full") is not None and v_design:
        # µAh * µV / 1e6 = µWh
        energy_full = to_int(attrs.get("charge_full")) * v_design // 1_000_000
        energy_design = (to_int(attrs.get("charge_full_design")) or 0) * v_design // 1_000_000 or None
        energy_now = (to_int(attrs.get("charge_now")) or 0) * v_design // 1_000_000
        current = to_int(attrs.get("current_now"))
        power_now = current * v_design // 1_000_000 if current is not None else None

    health = round(100 * energy_full / energy_design, 1) if energy_full and energy_design else None
    hours = None
    if power_now and power_now > 0 and energy_now is not None:
        if status == "Discharging":
            hours = energy_now / power_now
        elif status == "Charging" and energy_full:
            hours = max(0, energy_full - energy_now) / power_now
    return {
        "name": name,
        "status": status,
        "charge_percent": to_int(attrs.get("capacity")),
        "health_percent": health,
        "design_capacity_wh": round(energy_design / 1e6, 2) if energy_design else None,
        "full_charge_capacity_wh": round(energy_full / 1e6, 2) if energy_full else None,
        "current_energy_wh": round(energy_now / 1e6, 2) if energy_now is not None else None,
        "power_draw_w": round(power_now / 1e6, 2) if power_now else None,
        "estimated_time_remaining": _hours_to_text(hours),
        "estimate_kind": ("until empty" if status == "Discharging" else
                          "until full" if status == "Charging" else None) if hours else None,
        "cycle_count": to_int(attrs.get("cycle_count")) or None,
        "technology": clean(attrs.get("technology")),
        "manufacturer": clean(attrs.get("manufacturer")),
        "model": clean(attrs.get("model_name")),
    }


def parse_thermal_zones(zones: list[tuple[str, str | None, str | None]]) -> list[dict[str, Any]]:
    out = []
    for zone, ztype, temp in zones:
        milli = to_int(temp)
        if milli is None:
            continue
        out.append({"zone": zone, "sensor": clean(ztype), "temperature_c": round(milli / 1000, 1)})
    return out


# --------------------------------------------------------------------------- #
# OS
# --------------------------------------------------------------------------- #

def parse_os_release(text: str) -> dict[str, Any]:
    kv: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        kv[key.strip()] = value.strip().strip('"').strip("'")
    return {
        "pretty_name": kv.get("PRETTY_NAME"),
        "name": kv.get("NAME"),
        "version": kv.get("VERSION"),
        "version_id": kv.get("VERSION_ID"),
        "id": kv.get("ID"),
        "id_like": kv.get("ID_LIKE"),
        "codename": kv.get("VERSION_CODENAME"),
    }


def parse_uptime(text: str | None) -> dict[str, Any]:
    seconds = to_float((text or "").split()[0]) if text and text.split() else None
    if seconds is None:
        return {"uptime_seconds": None, "uptime": None}
    s = int(seconds)
    days, rem = divmod(s, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    human = (f"{days}d " if days else "") + f"{hours}h {minutes:02d}m"
    return {"uptime_seconds": s, "uptime": human}


def parse_kernel_version(text: str | None) -> str | None:
    m = re.match(r"Linux version (\S+)", text or "")
    return m.group(1) if m else None
