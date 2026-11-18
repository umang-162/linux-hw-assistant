"""Parser tests (stdlib unittest – no extra dependencies).

Run:  python -m unittest discover -s tests -v
"""

import json
import unittest

from src import parsers as P
from src.mock_data import get_fixture

LAPTOP = get_fixture("laptop")


def cmd(key, fixture=LAPTOP):
    return fixture["commands"][key]["stdout"]


class TestCpu(unittest.TestCase):
    def test_lscpu_tree_format(self):
        d = P.parse_lscpu(cmd("lscpu"))
        self.assertEqual(d["model_name"], "12th Gen Intel(R) Core(TM) i7-12700H")
        self.assertEqual(d["logical_cpus"], 20)
        self.assertEqual(d["physical_cores"], 14)
        self.assertEqual(d["max_mhz"], 4700.0)
        self.assertEqual(d["caches"]["L3"], {"total": "24 MiB", "instances": 1})
        self.assertIn("avx2", d["notable_flags"])

    def test_lscpu_legacy_flat_format(self):
        legacy = ("Architecture:          x86_64\nCPU(s):                8\n"
                  "Thread(s) per core:    2\nCore(s) per socket:    4\nSocket(s):             1\n"
                  "Model name:            Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz\n"
                  "CPU max MHz:           3400.0000\nL2 cache:              256K\n"
                  "L3 cache:              6144K\n")
        d = P.parse_lscpu(legacy)
        self.assertEqual(d["physical_cores"], 4)
        self.assertEqual(d["logical_cpus"], 8)
        self.assertEqual(d["caches"]["L3"]["total"], "6144K")

    def test_topology_detects_hybrid(self):
        t = P.parse_lscpu_topology(cmd("lscpu_topology"))
        self.assertEqual(t, {"logical_cpus": 20, "physical_cores": 14, "sockets": 1,
                             "cores_with_smt": 6, "cores_without_smt": 8,
                             "hybrid_core_layout": True})

    def test_proc_cpuinfo_fallback(self):
        d = P.parse_proc_cpuinfo(LAPTOP["files"]["/proc/cpuinfo"])
        self.assertEqual(d["logical_cpus"], 20)
        self.assertEqual(d["physical_cores"], 14)
        self.assertNotIn("hypervisor", d["notable_flags"])


class TestMemory(unittest.TestCase):
    def test_free_and_meminfo_agree(self):
        a = P.parse_free(cmd("free"))["ram"]
        b = P.parse_meminfo(LAPTOP["files"]["/proc/meminfo"])["ram"]
        self.assertEqual(a["total_gib"], b["total_gib"])
        self.assertEqual(a["available_gib"], b["available_gib"])
        self.assertEqual(a["used_gib"], b["used_gib"])

    def test_free_legacy_layout(self):
        legacy = ("             total       used       free     shared    buffers     cached\n"
                  "Mem:    8254439424 7000000000 1254439424  100000000  200000000 3000000000\n"
                  "-/+ buffers/cache: 3800000000 4454439424\n"
                  "Swap:   2147479552          0 2147479552\n")
        d = P.parse_free(legacy)
        self.assertEqual(d["ram"]["total_gib"], 7.69)
        self.assertEqual(d["ram"]["buff_cache_gib"], 2.98)

    def test_dmidecode_slots_and_upgrade(self):
        s = P.summarize_memory_slots(P.parse_dmidecode_memory(cmd("dmidecode_memory_sudo")))
        self.assertEqual((s["total_slots"], s["populated_slots"], s["empty_slots"]), (2, 1, 1))
        self.assertEqual(s["installed_gib"], 16.0)
        self.assertEqual(s["max_capacity_gib"], 64.0)
        m = s["modules"][0]
        self.assertEqual((m["type"], m["form_factor"], m["speed_mts"]), ("DDR5", "SODIMM", 4800))
        u = s["upgrade_analysis"]
        self.assertTrue(u["can_add_ram"])
        self.assertEqual(u["max_addable_without_replacing_gib"], 32.0)
        self.assertEqual(u["total_after_filling_empty_slots_gib"], 48.0)
        self.assertIn("2 x 32 GiB", u["to_reach_maximum"])

    def test_dmidecode_never_returns_serials(self):
        raw = cmd("dmidecode_memory_sudo")
        self.assertIn("4A7C19E2", raw)  # fixture contains a serial...
        out = json.dumps(P.summarize_memory_slots(P.parse_dmidecode_memory(raw)))
        self.assertNotIn("4A7C19E2", out)  # ...the parser must drop it
        self.assertNotIn("9876543210", out)

    def test_soldered_memory(self):
        text = ("Handle 0x0010, DMI type 16, 23 bytes\nPhysical Memory Array\n"
                "\tUse: System Memory\n\tMaximum Capacity: 32 GB\n\tNumber Of Devices: 2\n\n"
                "Handle 0x0011, DMI type 17, 92 bytes\nMemory Device\n\tArray Handle: 0x0010\n"
                "\tSize: 8 GB\n\tForm Factor: Row Of Chips\n\tLocator: ONBOARD\n\tType: DDR4\n"
                "\tSpeed: 3200 MT/s\n\n"
                "Handle 0x0012, DMI type 17, 92 bytes\nMemory Device\n\tArray Handle: 0x0010\n"
                "\tSize: No Module Installed\n\tLocator: DIMM 0\n\tType: Unknown\n")
        s = P.summarize_memory_slots(P.parse_dmidecode_memory(text))
        self.assertTrue(s["modules"][0]["soldered_onboard"])
        self.assertEqual(s["upgrade_analysis"]["max_addable_without_replacing_gib"], 24.0)
        self.assertTrue(any("soldered" in c for c in s["upgrade_analysis"]["caveats"]))

    def test_sizes(self):
        self.assertEqual(P.parse_mem_size_to_bytes("16 GB"), 16 * P.GIB)
        self.assertEqual(P.parse_mem_size_to_bytes("16384 MB"), 16 * P.GIB)
        self.assertEqual(P.parse_mem_size_to_bytes("11.5 MiB"), int(11.5 * P.MIB))
        self.assertIsNone(P.parse_mem_size_to_bytes("No Module Installed"))


def _phantom_laptop_dmi():
    """Lenovo-G-series style: firmware lists 4 devices, the board has 2 slots."""
    text = ("Handle 0x0005, DMI type 16, 23 bytes\nPhysical Memory Array\n"
            "\tUse: System Memory\n\tMaximum Capacity: 16 GB\n\tNumber Of Devices: 4\n\n")
    for i, loc in enumerate(["ChannelA-DIMM0", "DIMM1", "DIMM2", "DIMM3"]):
        size = "4096 MB" if i == 0 else "No Module Installed"
        text += (f"Handle 0x000{6 + i}, DMI type 17, 34 bytes\nMemory Device\n"
                 f"\tArray Handle: 0x0005\n\tSize: {size}\n\tForm Factor: SODIMM\n"
                 f"\tLocator: {loc}\n\tType: DDR3\n\tSpeed: 1600 MHz\n\n")
    return P.parse_dmidecode_memory(text)


class TestPhantomSlots(unittest.TestCase):
    def test_laptop_with_4_reported_slots_is_flagged(self):
        s = P.summarize_memory_slots(_phantom_laptop_dmi(), chassis_type="Notebook")
        self.assertEqual(s["total_slots"], 4)
        self.assertFalse(s["slot_count_reliable"])
        self.assertIn("phantom", s["upgrade_analysis"]["caveats"][0])

    def test_physical_slot_override_fixes_the_maths(self):
        s = P.summarize_memory_slots(_phantom_laptop_dmi(), physical_slots=2,
                                     chassis_type="Notebook")
        self.assertEqual((s["total_slots"], s["empty_slots"]), (2, 1))
        self.assertEqual(s["slots_reported_by_firmware"], 4)
        self.assertNotIn("slot_count_reliable", s)
        u = s["upgrade_analysis"]
        self.assertEqual(u["estimated_max_per_slot_gib"], 8.0)
        self.assertEqual(u["max_addable_without_replacing_gib"], 8.0)
        self.assertEqual(u["total_after_filling_empty_slots_gib"], 12.0)

    def test_desktop_with_4_slots_is_not_flagged(self):
        s = P.summarize_memory_slots(_phantom_laptop_dmi(), chassis_type="Desktop")
        self.assertNotIn("slot_count_reliable", s)


class TestStorage(unittest.TestCase):
    def test_lsblk_nvme_and_usb(self):
        d = P.parse_lsblk_json(cmd("lsblk"))
        disks = {x["name"]: x for x in d["disks"]}
        self.assertTrue(disks["nvme0n1"]["is_nvme"])
        self.assertEqual(disks["nvme0n1"]["media_type"], "NVMe SSD")
        self.assertEqual(disks["nvme0n1"]["size_gb_decimal"], 512.1)
        self.assertTrue(disks["sda"]["removable_or_usb"])
        self.assertEqual(d["loop_devices_skipped"], 2)
        self.assertIn("/media/<user>/USBSTICK", disks["sda"]["partitions"][0]["mountpoints"])

    def test_lsblk_old_string_format(self):
        old = json.dumps({"blockdevices": [
            {"name": "sda", "size": "1000204886016", "type": "disk", "mountpoint": None,
             "rota": "1", "tran": "sata", "model": "WDC WD10EZEX-08W"}]})
        disk = P.parse_lsblk_json(old)["disks"][0]
        self.assertEqual(disk["media_type"], "HDD (rotational)")
        self.assertEqual(disk["interface"], "sata")
        self.assertFalse(disk["is_nvme"])

    def test_df_redacts_username(self):
        rows = P.parse_df(cmd("df"))
        self.assertEqual(rows[0]["mountpoint"], "/")
        self.assertNotIn("alex", json.dumps(rows))


class TestPci(unittest.TestCase):
    def test_lspci_gpus(self):
        c = P.classify_pci(P.parse_lspci_vmm(cmd("lspci")))
        kinds = {g["vendor"]: g["kind"] for g in c["gpus"]}
        self.assertEqual(kinds["Intel Corporation"], "integrated")
        self.assertEqual(kinds["NVIDIA Corporation"], "discrete")
        nvidia = [g for g in c["gpus"] if "NVIDIA" in g["vendor"]][0]
        self.assertEqual(nvidia["kernel_modules"], ["nouveau", "nvidia_drm", "nvidia"])
        self.assertEqual(len(c["storage_controllers"]), 1)

    def test_sysfs_fallback(self):
        devs = P.parse_pci_sysfs([{"slot": "0000:01:00.0", "class_code": "0x030000",
                                   "vendor_id": "0x10de", "device_id": "0x25a2",
                                   "driver": "nvidia"}])
        self.assertEqual(devs[0]["class"], "VGA compatible controller")
        self.assertEqual(P.classify_pci(devs)["gpus"][0]["kind"], "discrete")

    def test_nvidia_smi(self):
        g = P.parse_nvidia_smi(cmd("nvidia_smi"))[0]
        self.assertEqual(g["vram_gib"], 4.0)
        self.assertEqual(g["driver_version"], "550.120")


class TestPowerAndOs(unittest.TestCase):
    def test_battery_energy_style(self):
        files = LAPTOP["files"]
        attrs = {k.rsplit("/", 1)[1]: v for k, v in files.items()
                 if k.startswith("/sys/class/power_supply/BAT0/")}
        b = P.parse_battery("BAT0", attrs)
        self.assertEqual(b["health_percent"], 89.5)
        self.assertEqual(b["estimated_time_remaining"], "3h 58m")

    def test_battery_charge_style(self):
        b = P.parse_battery("BAT1", {"status": "Charging", "charge_full": "4000000",
                                     "charge_full_design": "5000000", "charge_now": "2000000",
                                     "current_now": "1000000", "voltage_min_design": "11400000",
                                     "capacity": "50"})
        self.assertEqual(b["health_percent"], 80.0)
        self.assertEqual(b["estimate_kind"], "until full")

    def test_os_release(self):
        d = P.parse_os_release(LAPTOP["files"]["/etc/os-release"])
        self.assertEqual(d["pretty_name"], "Ubuntu 24.04.1 LTS")


if __name__ == "__main__":
    unittest.main()
