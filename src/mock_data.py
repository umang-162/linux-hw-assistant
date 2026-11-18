"""Realistic hardware fixtures for ``--mock`` mode.

Virtual machines / WSL2 hide physical DIMM slots, NVMe transports and real
GPUs, so this module reproduces the *exact raw text* that real tools print on
a physical Linux laptop. The parsers in :mod:`src.parsers` then run on that
text exactly as they would in production.

Reference machine (scenario ``laptop``)
---------------------------------------
* 12th Gen Intel Core i7-12700H — 14 cores (6 with SMT + 8 without), 20 threads
* 16 GB DDR5-4800 SODIMM in **1 of 2** slots, firmware-reported max **64 GB**
* 512 GB Samsung NVMe SSD (+ a USB flash drive plugged in)
* Intel Iris Xe iGPU + NVIDIA GeForce RTX 3050 Laptop GPU (4 GB)
* Ubuntu 24.04.1 LTS, kernel 6.8, UEFI, battery at 78 % and discharging

Scenarios
---------
``laptop``         sudo -n works (passwordless dmidecode rule configured)
``laptop-nosudo``  same machine, but ``sudo -n`` needs a password
``wsl2``           running inside WSL2 – no SMBIOS, virtual disks, no battery
``minimal``        bare metal but lscpu / lspci / lsblk / nvidia-smi missing,
                   forcing the ``/proc`` and ``/sys`` fallbacks

The vendor/product strings are deliberately generic ("ExampleCorp") so the
fixture is never mistaken for a claim about a specific real laptop model.
"""

from __future__ import annotations

import copy
import json
from typing import Any

# --------------------------------------------------------------------------- #
# CPU
# --------------------------------------------------------------------------- #

_CPU_FLAGS = (
    "fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush "
    "dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc "
    "art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf "
    "tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma "
    "cx16 xtpr pdcm sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx "
    "f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp "
    "ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 "
    "smep bmi2 erms invpcid rdseed adx smap clflushopt clwb intel_pt sha_ni xsaveopt "
    "xsavec xgetbv1 xsaves split_lock_detect avx_vnni dtherm ida arat pln pts hwp "
    "hwp_notify hwp_act_window hwp_epp hwp_pkg_req hfi vnmi umip pku ospke waitpkg gfni "
    "vaes vpclmulqdq rdpid movdiri movdir64b fsrm md_clear serialize arch_lbr ibt "
    "flush_l1d arch_capabilities"
)

_LSCPU_LAPTOP = """\
Architecture:                         x86_64
  CPU op-mode(s):                     32-bit, 64-bit
  Address sizes:                      39 bits physical, 48 bits virtual
  Byte Order:                         Little Endian
CPU(s):                               20
  On-line CPU(s) list:                0-19
Vendor ID:                            GenuineIntel
  Model name:                         12th Gen Intel(R) Core(TM) i7-12700H
    CPU family:                       6
    Model:                            154
    Thread(s) per core:               2
    Core(s) per socket:               14
    Socket(s):                        1
    Stepping:                         3
    CPU(s) scaling MHz:               21%
    CPU max MHz:                      4700.0000
    CPU min MHz:                      400.0000
    BogoMIPS:                         5376.00
    Flags:                            {flags}
Virtualization features:
  Virtualization:                     VT-x
Caches (sum of all):
  L1d:                                544 KiB (14 instances)
  L1i:                                704 KiB (14 instances)
  L2:                                 11.5 MiB (8 instances)
  L3:                                 24 MiB (1 instance)
NUMA:
  NUMA node(s):                       1
  NUMA node0 CPU(s):                  0-19
Vulnerabilities:
  Gather data sampling:               Not affected
  Itlb multihit:                      Not affected
  L1tf:                               Not affected
  Mds:                                Not affected
  Meltdown:                           Not affected
  Spectre v1:                         Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:                         Mitigation; Enhanced / Automatic IBRS; IBPB conditional; RSB filling; PBRSB-eIBRS SW sequence; BHI BHI_DIS_S
""".format(flags=_CPU_FLAGS)

# 6 performance cores with 2 threads each (CPUs 0-11) + 8 efficiency cores
# with 1 thread each (CPUs 12-19).
_TOPOLOGY = [(cpu, cpu // 2) for cpu in range(12)] + [(12 + i, 6 + i) for i in range(8)]


def _lscpu_topology() -> str:
    lines = [
        "# The following is the parsable format, which can be fed to other",
        "# programs. Each different item in every column has an unique ID",
        "# starting usually from zero.",
        "# CPU,Core,Socket,Node",
    ]
    lines += [f"{cpu},{core},0,0" for cpu, core in _TOPOLOGY]
    return "\n".join(lines) + "\n"


def _proc_cpuinfo(hypervisor: bool = False, model: str | None = None,
                  n_cpus: int | None = None) -> str:
    flags = _CPU_FLAGS + (" hypervisor" if hypervisor else "")
    model = model or "12th Gen Intel(R) Core(TM) i7-12700H"
    topo = _TOPOLOGY if n_cpus is None else [(i, i // 2) for i in range(n_cpus)]
    cores = len({c for _, c in topo})
    blocks = []
    for cpu, core in topo:
        # Real kernels number E-cores' core ids non-contiguously; mimic that.
        core_id = core * 4 if core < 6 else 24 + (core - 6)
        mhz = "2700.000" if core < 6 else "2100.000"
        blocks.append(
            f"processor\t: {cpu}\n"
            "vendor_id\t: GenuineIntel\n"
            "cpu family\t: 6\n"
            "model\t\t: 154\n"
            f"model name\t: {model}\n"
            "stepping\t: 3\n"
            "microcode\t: 0x4121\n"
            f"cpu MHz\t\t: {mhz}\n"
            "cache size\t: 24576 KB\n"
            "physical id\t: 0\n"
            f"siblings\t: {len(topo)}\n"
            f"core id\t\t: {core_id}\n"
            f"cpu cores\t: {cores}\n"
            f"apicid\t\t: {cpu}\n"
            "fpu\t\t: yes\n"
            "fpu_exception\t: yes\n"
            "cpuid level\t: 32\n"
            "wp\t\t: yes\n"
            f"flags\t\t: {flags}\n"
            "bogomips\t: 5376.00\n"
            "clflush size\t: 64\n"
            "cache_alignment\t: 64\n"
            "address sizes\t: 39 bits physical, 48 bits virtual\n"
            "power management:\n"
        )
    return "\n".join(blocks) + "\n"


def _cpufreq_files() -> dict[str, str]:
    files: dict[str, str] = {}
    for cpu, core in _TOPOLOGY:
        base = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/"
        files[base + "cpuinfo_max_freq"] = "4700000\n" if core < 6 else "3500000\n"
        files[base + "cpuinfo_min_freq"] = "400000\n"
        files[base + "scaling_governor"] = "powersave\n"
        files[base + "scaling_driver"] = "intel_pstate\n"
        files[base + "energy_performance_preference"] = "balance_performance\n"
    return files


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #

def _meminfo(total_kb: int, free_kb: int, avail_kb: int, buffers_kb: int,
             cached_kb: int, sreclaim_kb: int, shmem_kb: int,
             swap_total_kb: int, swap_free_kb: int) -> str:
    rows = [
        ("MemTotal", total_kb), ("MemFree", free_kb), ("MemAvailable", avail_kb),
        ("Buffers", buffers_kb), ("Cached", cached_kb), ("SwapCached", 0),
        ("Active", 4123456), ("Inactive", 3567890), ("SwapTotal", swap_total_kb),
        ("SwapFree", swap_free_kb), ("Dirty", 1204), ("Shmem", shmem_kb),
        ("SReclaimable", sreclaim_kb), ("SUnreclaim", 201344),
        ("HugePages_Total", None), ("Hugepagesize", 2048),
    ]
    lines = []
    for key, val in rows:
        if key == "HugePages_Total":
            lines.append(f"{'HugePages_Total:':<16}{0:>8}")
        else:
            lines.append(f"{key + ':':<16}{val:>8} kB")
    return "\n".join(lines) + "\n"


def _free_b(total_kb: int, free_kb: int, avail_kb: int, buffers_kb: int,
            cached_kb: int, sreclaim_kb: int, shmem_kb: int,
            swap_total_kb: int, swap_free_kb: int) -> str:
    """Render ``free -b`` exactly as procps-ng computes it from /proc/meminfo."""
    b = 1024
    buff_cache = (buffers_kb + cached_kb + sreclaim_kb) * b
    total = total_kb * b
    free = free_kb * b
    used = total - free - buff_cache
    swap_total, swap_free = swap_total_kb * b, swap_free_kb * b
    header = f"{'':15}{'total':>13}{'used':>13}{'free':>13}{'shared':>13}{'buff/cache':>13}{'available':>13}"
    mem = (f"{'Mem:':15}{total:>13}{used:>13}{free:>13}{shmem_kb * b:>13}"
           f"{buff_cache:>13}{avail_kb * b:>13}")
    swap = f"{'Swap:':15}{swap_total:>13}{swap_total - swap_free:>13}{swap_free:>13}"
    return "\n".join([header, mem, swap]) + "\n"


_LAPTOP_MEM = dict(total_kb=15994812, free_kb=3421304, avail_kb=9312456,
                   buffers_kb=412340, cached_kb=5824116, sreclaim_kb=402112,
                   shmem_kb=812344, swap_total_kb=4194300, swap_free_kb=4194300)

_WSL_MEM = dict(total_kb=8038232, free_kb=6102344, avail_kb=7412876,
                buffers_kb=81232, cached_kb=1134566, sreclaim_kb=98112,
                shmem_kb=3204, swap_total_kb=2097152, swap_free_kb=2097152)

_DMIDECODE_LAPTOP = """\
# dmidecode 3.5
Getting SMBIOS data from sysfs.
SMBIOS 3.4.0 present.

Handle 0x0022, DMI type 16, 23 bytes
Physical Memory Array
\tLocation: System Board Or Motherboard
\tUse: System Memory
\tError Correction Type: None
\tMaximum Capacity: 64 GB
\tError Information Handle: Not Provided
\tNumber Of Devices: 2

Handle 0x0023, DMI type 17, 92 bytes
Memory Device
\tArray Handle: 0x0022
\tError Information Handle: Not Provided
\tTotal Width: 64 bits
\tData Width: 64 bits
\tSize: 16 GB
\tForm Factor: SODIMM
\tSet: None
\tLocator: Controller0-ChannelA-DIMM0
\tBank Locator: BANK 0
\tType: DDR5
\tType Detail: Synchronous
\tSpeed: 4800 MT/s
\tManufacturer: Samsung
\tSerial Number: 4A7C19E2
\tAsset Tag: 9876543210
\tPart Number: M425R2GA3BB0-CQKOD
\tRank: 1
\tConfigured Memory Speed: 4800 MT/s
\tMinimum Voltage: 1.1 V
\tMaximum Voltage: 1.1 V
\tConfigured Voltage: 1.1 V
\tMemory Technology: DRAM
\tMemory Operating Mode Capability: Volatile memory
\tFirmware Version: Not Specified
\tModule Manufacturer ID: Bank 1, Hex 0xCE
\tModule Product ID: Unknown
\tMemory Subsystem Controller Manufacturer ID: Unknown
\tMemory Subsystem Controller Product ID: Unknown
\tNon-Volatile Size: None
\tVolatile Size: 16 GB
\tCache Size: None
\tLogical Size: None

Handle 0x0024, DMI type 17, 92 bytes
Memory Device
\tArray Handle: 0x0022
\tError Information Handle: Not Provided
\tTotal Width: Unknown
\tData Width: Unknown
\tSize: No Module Installed
\tForm Factor: Unknown
\tSet: None
\tLocator: Controller1-ChannelA-DIMM0
\tBank Locator: BANK 0
\tType: Unknown
\tType Detail: None
\tSpeed: Unknown
\tManufacturer: Not Specified
\tSerial Number: Not Specified
\tAsset Tag: Not Specified
\tPart Number: Not Specified
\tRank: Unknown
\tConfigured Memory Speed: Unknown
\tMinimum Voltage: Unknown
\tMaximum Voltage: Unknown
\tConfigured Voltage: Unknown
\tMemory Technology: Unknown
\tMemory Operating Mode Capability: Unknown
\tFirmware Version: Unknown
\tModule Manufacturer ID: Unknown
\tModule Product ID: Unknown
\tMemory Subsystem Controller Manufacturer ID: Unknown
\tMemory Subsystem Controller Product ID: Unknown
\tNon-Volatile Size: None
\tVolatile Size: None
\tCache Size: None
\tLogical Size: None
"""

_DMI_ID_LAPTOP = {
    "sys_vendor": "ExampleCorp",
    "product_name": "ExampleBook Pro 15 (2022)",
    "product_version": "EB15-G2",
    "product_family": "ExampleBook",
    "board_vendor": "ExampleCorp",
    "board_name": "EB15-MB-ADL",
    "board_version": "1.0",
    "bios_vendor": "American Megatrends International, LLC.",
    "bios_version": "EB15.312",
    "bios_date": "03/14/2024",
    "chassis_type": "10",
    "chassis_vendor": "ExampleCorp",
}

# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #

_LSBLK_LAPTOP = {
    "blockdevices": [
        {"name": "loop0", "size": 4096, "type": "loop", "mountpoint": "/snap/bare/5",
         "rota": False, "tran": None, "model": None},
        {"name": "loop1", "size": 77492224, "type": "loop", "mountpoint": "/snap/core22/1663",
         "rota": False, "tran": None, "model": None},
        {"name": "sda", "size": 31914983424, "type": "disk", "mountpoint": None,
         "rota": False, "tran": "usb", "model": "SanDisk 3.2Gen1",
         "children": [
             {"name": "sda1", "size": 31913934848, "type": "part",
              "mountpoint": "/media/alex/USBSTICK", "rota": False, "tran": None, "model": None},
         ]},
        {"name": "zram0", "size": 4294967296, "type": "disk", "mountpoint": "[SWAP]",
         "rota": False, "tran": None, "model": None},
        {"name": "nvme0n1", "size": 512110190592, "type": "disk", "mountpoint": None,
         "rota": False, "tran": "nvme", "model": "SAMSUNG MZVL2512HCJQ-00B00",
         "children": [
             {"name": "nvme0n1p1", "size": 536870912, "type": "part",
              "mountpoint": "/boot/efi", "rota": False, "tran": "nvme", "model": None},
             {"name": "nvme0n1p2", "size": 511571771392, "type": "part",
              "mountpoint": "/", "rota": False, "tran": "nvme", "model": None},
         ]},
    ]
}

_DF_LAPTOP = """\
Filesystem     Type      1B-blocks         Used        Avail Mounted on
/dev/nvme0n1p2 ext4   502921060352 187345678336 289936658432 /
/dev/nvme0n1p1 vfat      535805952      6291456    529514496 /boot/efi
/dev/sda1      exfat   31902400512   8589934592  23312465920 /media/alex/USBSTICK
"""

# --------------------------------------------------------------------------- #
# PCI
# --------------------------------------------------------------------------- #

# (slot, class, vendor, device, svendor, sdevice, rev, driver, modules,
#  class_code, vendor_id, device_id)
_PCI_LAPTOP = [
    ("00:00.0", "Host bridge", "Intel Corporation",
     "12th Gen Core Processor Host Bridge/DRAM Registers", "ExampleCorp", "Device 3b1a",
     "02", "igen6_edac", ["igen6_edac"], "0x060000", "0x8086", "0x4621"),
    ("00:02.0", "VGA compatible controller", "Intel Corporation",
     "Alder Lake-P GT2 [Iris Xe Graphics]", "ExampleCorp", "Device 3b1a",
     "0c", "i915", ["i915"], "0x030000", "0x8086", "0x46a6"),
    ("00:14.0", "USB controller", "Intel Corporation",
     "Alder Lake PCH USB 3.2 xHCI Host Controller", "ExampleCorp", "Device 3b1a",
     "01", "xhci_hcd", ["xhci_pci"], "0x0c0330", "0x8086", "0x51ed"),
    ("00:14.3", "Network controller", "Intel Corporation",
     "Alder Lake-P PCH CNVi WiFi", "Intel Corporation", "Wi-Fi 6 AX201 160MHz",
     "01", "iwlwifi", ["iwlwifi"], "0x028000", "0x8086", "0x51f0"),
    ("00:1f.3", "Audio device", "Intel Corporation",
     "Alder Lake PCH-P High Definition Audio Controller", "ExampleCorp", "Device 3b1a",
     "01", "sof-audio-pci-intel-tgl", ["snd_hda_intel", "snd_sof_pci_intel_tgl"],
     "0x040380", "0x8086", "0x51c8"),
    ("01:00.0", "VGA compatible controller", "NVIDIA Corporation",
     "GA107M [GeForce RTX 3050 Mobile]", "ExampleCorp", "Device 3b1a",
     "a1", "nvidia", ["nouveau", "nvidia_drm", "nvidia"], "0x030000", "0x10de", "0x25a2"),
    ("03:00.0", "Ethernet controller", "Realtek Semiconductor Co., Ltd.",
     "RTL8111/8168/8211/8411 PCI Express Gigabit Ethernet Controller", "ExampleCorp",
     "Device 3b1a", "15", "r8169", ["r8169"], "0x020000", "0x10ec", "0x8168"),
    ("04:00.0", "Non-Volatile memory controller", "Samsung Electronics Co Ltd",
     "NVMe SSD Controller PM9A1/PM9A3/980PRO", "Samsung Electronics Co Ltd",
     "SSD 980 PRO", "00", "nvme", ["nvme"], "0x010802", "0x144d", "0xa80a"),
]


def _lspci_vmm(devices: list[tuple]) -> str:
    blocks = []
    for (slot, cls, vendor, device, svendor, sdevice, rev, driver, modules,
         *_ids) in devices:
        lines = [f"Slot:\t{slot}", f"Class:\t{cls}", f"Vendor:\t{vendor}", f"Device:\t{device}"]
        if svendor:
            lines += [f"SVendor:\t{svendor}", f"SDevice:\t{sdevice}"]
        if rev:
            lines.append(f"Rev:\t{rev}")
        if driver:
            lines.append(f"Driver:\t{driver}")
        lines += [f"Module:\t{m}" for m in modules]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n\n"


def _pci_sysfs(devices: list[tuple]) -> tuple[dict[str, str], dict[str, str]]:
    files: dict[str, str] = {}
    links: dict[str, str] = {}
    for (slot, _cls, _v, _d, _sv, _sd, rev, driver, _mods,
         class_code, vendor_id, device_id) in devices:
        base = f"/sys/bus/pci/devices/0000:{slot}/"
        files[base + "class"] = class_code + "\n"
        files[base + "vendor"] = vendor_id + "\n"
        files[base + "device"] = device_id + "\n"
        files[base + "revision"] = f"0x{rev}\n"
        if driver:
            links[base + "driver"] = f"../../../../bus/pci/drivers/{driver}"
    return files, links


# --------------------------------------------------------------------------- #
# Power, thermal, OS
# --------------------------------------------------------------------------- #

_POWER_LAPTOP = {
    "/sys/class/power_supply/BAT0/type": "Battery",
    "/sys/class/power_supply/BAT0/present": "1",
    "/sys/class/power_supply/BAT0/status": "Discharging",
    "/sys/class/power_supply/BAT0/capacity": "78",
    "/sys/class/power_supply/BAT0/capacity_level": "Normal",
    "/sys/class/power_supply/BAT0/technology": "Li-poly",
    "/sys/class/power_supply/BAT0/manufacturer": "ExampleCell",
    "/sys/class/power_supply/BAT0/model_name": "EB15-56WH",
    "/sys/class/power_supply/BAT0/cycle_count": "214",
    "/sys/class/power_supply/BAT0/energy_full_design": "56000000",
    "/sys/class/power_supply/BAT0/energy_full": "50120000",
    "/sys/class/power_supply/BAT0/energy_now": "39093600",
    "/sys/class/power_supply/BAT0/power_now": "9850000",
    "/sys/class/power_supply/BAT0/voltage_now": "15912000",
    "/sys/class/power_supply/BAT0/voltage_min_design": "15480000",
    "/sys/class/power_supply/AC/type": "Mains",
    "/sys/class/power_supply/AC/online": "0",
}

_THERMAL_LAPTOP = {
    "/sys/class/thermal/thermal_zone0/type": "acpitz",
    "/sys/class/thermal/thermal_zone0/temp": "48000",
    "/sys/class/thermal/thermal_zone1/type": "INT3400 Thermal",
    "/sys/class/thermal/thermal_zone1/temp": "20000",
    "/sys/class/thermal/thermal_zone2/type": "TCPU",
    "/sys/class/thermal/thermal_zone2/temp": "58050",
    "/sys/class/thermal/thermal_zone3/type": "x86_pkg_temp",
    "/sys/class/thermal/thermal_zone3/temp": "61000",
}

_OS_RELEASE_UBUNTU = """\
PRETTY_NAME="Ubuntu 24.04.1 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.1 LTS (Noble Numbat)"
VERSION_CODENAME=noble
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
UBUNTU_CODENAME=noble
"""


def _block_sysfs_laptop() -> dict[str, str]:
    """``/sys/block`` view used when ``lsblk`` is missing."""
    return {
        "/sys/block/nvme0n1/size": str(512110190592 // 512),
        "/sys/block/nvme0n1/removable": "0",
        "/sys/block/nvme0n1/queue/rotational": "0",
        "/sys/block/nvme0n1/device/model": "SAMSUNG MZVL2512HCJQ-00B00",
        "/sys/block/sda/size": str(31914983424 // 512),
        "/sys/block/sda/removable": "1",
        "/sys/block/sda/queue/rotational": "0",
        "/sys/block/sda/device/model": "SanDisk 3.2Gen1",
        "/sys/block/loop0/size": "8",
        "/sys/block/loop0/removable": "0",
        "/sys/block/loop0/queue/rotational": "0",
        "/sys/block/zram0/size": str(4294967296 // 512),
        "/sys/block/zram0/removable": "0",
        "/sys/block/zram0/queue/rotational": "0",
    }


def _block_links_laptop() -> dict[str, str]:
    return {
        "/sys/block/nvme0n1": "../devices/pci0000:00/0000:00:06.0/0000:04:00.0/nvme/nvme0/nvme0n1",
        "/sys/block/sda": "../devices/pci0000:00/0000:00:14.0/usb2/2-1/2-1:1.0/host0/target0:0:0/0:0:0:0/block/sda",
        "/sys/block/loop0": "../devices/virtual/block/loop0",
        "/sys/block/zram0": "../devices/virtual/block/zram0",
    }


_BINARIES_FULL = {
    "lscpu": "/usr/bin/lscpu",
    "free": "/usr/bin/free",
    "sudo": "/usr/bin/sudo",
    "dmidecode": "/usr/sbin/dmidecode",
    "lsblk": "/usr/bin/lsblk",
    "df": "/usr/bin/df",
    "lspci": "/usr/bin/lspci",
    "nvidia-smi": "/usr/bin/nvidia-smi",
    "systemd-detect-virt": "/usr/bin/systemd-detect-virt",
    "uname": "/usr/bin/uname",
}


def _laptop() -> dict[str, Any]:
    pci_files, pci_links = _pci_sysfs(_PCI_LAPTOP)
    files: dict[str, str] = {
        "/proc/cpuinfo": _proc_cpuinfo(),
        "/proc/meminfo": _meminfo(**_LAPTOP_MEM),
        "/proc/version": ("Linux version 6.8.0-48-generic (buildd@lcy02-amd64-010) "
                          "(x86_64-linux-gnu-gcc-13 (Ubuntu 13.2.0-23ubuntu4) 13.2.0, GNU ld "
                          "(GNU Binutils for Ubuntu) 2.42) #48-Ubuntu SMP PREEMPT_DYNAMIC "
                          "Fri Sep 27 14:04:52 UTC 2024\n"),
        "/proc/uptime": "41873.22 512334.10\n",
        "/proc/sys/kernel/osrelease": "6.8.0-48-generic\n",
        "/proc/sys/kernel/ostype": "Linux\n",
        "/etc/os-release": _OS_RELEASE_UBUNTU,
        "/sys/firmware/efi/fw_platform_size": "64\n",
        "/sys/firmware/acpi/platform_profile": "balanced\n",
        "/sys/firmware/acpi/platform_profile_choices": "low-power balanced performance\n",
    }
    files.update({f"/sys/devices/virtual/dmi/id/{k}": v + "\n" for k, v in _DMI_ID_LAPTOP.items()})
    # Root-only identifiers exist on real systems; the executor's deny-list
    # must stop them from ever being read, which the tests verify.
    files["/sys/devices/virtual/dmi/id/product_serial"] = "PF3XAMPL3\n"
    files["/sys/devices/virtual/dmi/id/product_uuid"] = "4c4c4544-0000-1111-2222-333344445555\n"
    files.update(_cpufreq_files())
    files.update(pci_files)
    files.update(_POWER_LAPTOP)
    files.update(_THERMAL_LAPTOP)
    files.update(_block_sysfs_laptop())
    files.update({
        "/sys/class/drm/card1/device/vendor": "0x8086\n",
        "/sys/class/drm/card2/device/vendor": "0x10de\n",
    })
    links = dict(pci_links)
    links.update(_block_links_laptop())

    return {
        "scenario": "laptop",
        "description": "Bare-metal laptop, passwordless `sudo -n dmidecode` configured",
        "root": False,
        "binaries": dict(_BINARIES_FULL),
        "files": files,
        "links": links,
        "commands": {
            "lscpu": {"stdout": _LSCPU_LAPTOP},
            "lscpu_topology": {"stdout": _lscpu_topology()},
            "free": {"stdout": _free_b(**_LAPTOP_MEM)},
            "dmidecode_memory_sudo": {"stdout": _DMIDECODE_LAPTOP},
            "dmidecode_memory_root": {"rc": 1, "stderr": (
                "/sys/firmware/dmi/tables/smbios_entry_point: Permission denied\n"
                "Scanning /dev/mem for entry point.\n/dev/mem: Permission denied\n")},
            "lsblk": {"stdout": json.dumps(_LSBLK_LAPTOP, indent=3) + "\n"},
            "df": {"stdout": _DF_LAPTOP},
            "lspci": {"stdout": _lspci_vmm(_PCI_LAPTOP)},
            "nvidia_smi": {"stdout": "NVIDIA GeForce RTX 3050 Laptop GPU, 4096, 550.120, 45, 7.12, 0\n"},
            "systemd_detect_virt": {"stdout": "none\n", "rc": 1},
            "uname_machine": {"stdout": "x86_64\n"},
        },
    }


def _laptop_nosudo() -> dict[str, Any]:
    fx = _laptop()
    fx["scenario"] = "laptop-nosudo"
    fx["description"] = "Same laptop, but `sudo -n` needs a password (default Ubuntu setup)"
    fx["commands"]["dmidecode_memory_sudo"] = {
        "rc": 1, "stderr": "sudo: a password is required\n"}
    return fx


def _minimal() -> dict[str, Any]:
    fx = _laptop()
    fx["scenario"] = "minimal"
    fx["description"] = ("Bare-metal laptop on a minimal install: lscpu, lspci, lsblk and "
                         "nvidia-smi are missing, so /proc and /sys fallbacks are used")
    for b in ("lscpu", "lspci", "lsblk", "nvidia-smi"):
        fx["binaries"].pop(b, None)
    for k in ("lscpu", "lscpu_topology", "lspci", "lsblk", "nvidia_smi"):
        fx["commands"].pop(k, None)
    fx["commands"]["dmidecode_memory_sudo"] = {"rc": 1, "stderr": "sudo: a password is required\n"}
    return fx


def _wsl2() -> dict[str, Any]:
    lscpu = """\
Architecture:                         x86_64
  CPU op-mode(s):                     32-bit, 64-bit
  Address sizes:                      39 bits physical, 48 bits virtual
  Byte Order:                         Little Endian
CPU(s):                               20
  On-line CPU(s) list:                0-19
Vendor ID:                            GenuineIntel
  Model name:                         12th Gen Intel(R) Core(TM) i7-12700H
    CPU family:                       6
    Model:                            154
    Thread(s) per core:               2
    Core(s) per socket:               10
    Socket(s):                        1
    Stepping:                         3
    BogoMIPS:                         5375.99
    Flags:                            {flags} hypervisor
Virtualization features:
  Virtualization:                     VT-x
  Hypervisor vendor:                  Microsoft
  Virtualization type:                full
Caches (sum of all):
  L1d:                                480 KiB (10 instances)
  L1i:                                320 KiB (10 instances)
  L2:                                 12.5 MiB (10 instances)
  L3:                                 24 MiB (1 instance)
""".format(flags=_CPU_FLAGS)
    topo = "\n".join(
        ["# CPU,Core,Socket,Node"] + [f"{c},{c // 2},0," for c in range(20)]) + "\n"
    lsblk = {"blockdevices": [
        {"name": "sda", "size": 408883200, "type": "disk", "mountpoint": None,
         "rota": True, "tran": None, "model": "Virtual Disk"},
        {"name": "sdb", "size": 2147487744, "type": "disk", "mountpoint": "[SWAP]",
         "rota": True, "tran": None, "model": "Virtual Disk"},
        {"name": "sdc", "size": 1099511627776, "type": "disk", "mountpoint": "/",
         "rota": True, "tran": None, "model": "Virtual Disk"},
    ]}
    df = """\
Filesystem     Type     1B-blocks        Used         Avail Mounted on
/dev/sdc       ext4  1081101176832 21474836480 1004602810368 /
C:\\            9p     511571771392 301234567168  210337204224 /mnt/c
"""
    wsl_pci = [
        ("5582:00:00.0", "3D controller", "Microsoft Corporation", "Basic Render Driver",
         "", "", "", "", [], "0x030200", "0x1414", "0x008e"),
        ("f69c:00:00.0", "SCSI storage controller", "Red Hat, Inc.", "Virtio 1.0 filesystem",
         "Red Hat, Inc.", "Device 1100", "01", "virtio-pci", ["virtio_pci"],
         "0x010000", "0x1af4", "0x105a"),
    ]
    files = {
        "/proc/cpuinfo": _proc_cpuinfo(hypervisor=True, n_cpus=20),
        "/proc/meminfo": _meminfo(**_WSL_MEM),
        "/proc/version": ("Linux version 5.15.167.4-microsoft-standard-WSL2 "
                          "(root@f9c826d3017f) (gcc (GCC) 11.2.0, GNU ld (GNU Binutils) 2.37) "
                          "#1 SMP Tue Nov 5 00:21:55 UTC 2024\n"),
        "/proc/uptime": "1834.51 36102.88\n",
        "/proc/sys/kernel/osrelease": "5.15.167.4-microsoft-standard-WSL2\n",
        "/proc/sys/kernel/ostype": "Linux\n",
        "/etc/os-release": _OS_RELEASE_UBUNTU,
    }
    pci_files, pci_links = _pci_sysfs(wsl_pci)
    files.update(pci_files)
    binaries = dict(_BINARIES_FULL)
    binaries.pop("nvidia-smi")
    return {
        "scenario": "wsl2",
        "description": "Ubuntu inside WSL2 on Windows 11 – no SMBIOS, virtual disks/GPU, no battery",
        "root": False,
        "binaries": binaries,
        "files": files,
        "links": pci_links,
        "commands": {
            "lscpu": {"stdout": lscpu},
            "lscpu_topology": {"stdout": topo},
            "free": {"stdout": _free_b(**_WSL_MEM)},
            # Passwordless sudo succeeds, but WSL exposes no SMBIOS tables.
            "dmidecode_memory_sudo": {"stdout": "# dmidecode 3.5\n# No SMBIOS nor DMI entry point found, sorry.\n"},
            "dmidecode_memory_root": {"rc": 1, "stderr": "/dev/mem: Permission denied\n"},
            "lsblk": {"stdout": json.dumps(lsblk, indent=3) + "\n"},
            "df": {"stdout": df},
            "lspci": {"stdout": _lspci_vmm(wsl_pci)},
            "systemd_detect_virt": {"stdout": "wsl\n", "rc": 0},
            "uname_machine": {"stdout": "x86_64\n"},
        },
    }


_BUILDERS = {
    "laptop": _laptop,
    "laptop-nosudo": _laptop_nosudo,
    "wsl2": _wsl2,
    "minimal": _minimal,
}


def available_scenarios() -> dict[str, str]:
    return {name: build()["description"] for name, build in _BUILDERS.items()}


def get_fixture(scenario: str = "laptop") -> dict[str, Any]:
    try:
        return copy.deepcopy(_BUILDERS[scenario]())
    except KeyError as exc:
        raise ValueError(f"Unknown mock scenario {scenario!r}; "
                         f"choose from {', '.join(_BUILDERS)}") from exc
