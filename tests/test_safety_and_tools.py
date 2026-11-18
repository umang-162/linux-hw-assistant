"""Security-policy, tool-envelope and agent-loop tests (no network, no API key)."""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from src import executor as X
from src.agent import HardwareAgent
from src.config import Settings
from src.executor import LiveSource, MockSource, build_source, is_path_allowed
from src.mock_data import get_fixture
from src.tools import ToolRegistry


class TestExecutorPolicy(unittest.TestCase):
    def test_unknown_command_key_is_blocked(self):
        res = MockSource(get_fixture("laptop")).run("rm -rf /")
        self.assertFalse(res.ok)
        self.assertEqual(res.error_kind, "not_allowed")

    def test_allow_list_contains_no_shell_or_writes(self):
        shells_and_writers = {"sh", "bash", "zsh", "dash", "env", "rm", "dd", "mkfs", "tee",
                              "chmod", "chown", "mv", "cp", "python", "python3", "perl"}
        for key, argv in X.ALLOWED_COMMANDS.items():
            self.assertNotIn(argv[0], shells_and_writers, key)
            for token in argv:
                for meta in (";", "|", "&", ">", "<", "`", "$(", "\n"):
                    self.assertNotIn(meta, token, f"{key}: {token}")
            if argv[0] == "sudo":
                self.assertEqual(argv[1], "-n", "sudo must be non-interactive")

    def test_path_policy(self):
        self.assertTrue(is_path_allowed("/proc/meminfo"))
        self.assertTrue(is_path_allowed("/sys/class/power_supply/BAT0/capacity"))
        self.assertFalse(is_path_allowed("/etc/shadow"))
        self.assertFalse(is_path_allowed("/proc/1/environ"))
        self.assertFalse(is_path_allowed("/sys/devices/virtual/dmi/id/product_serial"))
        self.assertFalse(is_path_allowed("/sys/devices/virtual/dmi/id/product_uuid"))
        self.assertFalse(is_path_allowed("/sys/class/power_supply/../../../etc/shadow"))
        self.assertFalse(is_path_allowed("relative/path"))

    def test_serial_not_readable_even_if_present(self):
        src = MockSource(get_fixture("laptop"))
        self.assertIsNone(src.read_value("/sys/devices/virtual/dmi/id/product_serial"))

    def test_live_runner_uses_shell_false_and_timeout(self):
        src = LiveSource(timeout=5)
        fake = SimpleNamespace(returncode=0, stdout="x86_64\n", stderr="")
        with mock.patch.object(src, "which", return_value="/usr/bin/uname"), \
                mock.patch("src.executor.subprocess.run", return_value=fake) as run:
            res = src.run("uname_machine")
        self.assertTrue(res.ok)
        args, kwargs = run.call_args
        self.assertIsInstance(args[0], list)
        self.assertIs(kwargs["shell"], False)
        self.assertEqual(kwargs["timeout"], 5)
        self.assertTrue(kwargs["capture_output"])
        self.assertTrue(kwargs["text"])

    def test_failure_classification(self):
        self.assertEqual(X.classify_failure("", "sudo: a password is required"),
                         "permission_denied")
        self.assertEqual(X.classify_failure("# No SMBIOS nor DMI entry point found, sorry.", ""),
                         "no_smbios")


class TestToolsAcrossScenarios(unittest.TestCase):
    def test_laptop_ram_ok(self):
        out = ToolRegistry(build_source(True, "laptop")).dispatch("inspect_ram_slots_and_capacity")
        self.assertEqual(out.status, "ok")
        self.assertEqual(out.payload["data"]["upgrade_analysis"]["headroom_gib"], 48.0)

    def test_nosudo_reports_permission_required(self):
        out = ToolRegistry(build_source(True, "laptop-nosudo")).dispatch(
            "inspect_ram_slots_and_capacity")
        self.assertEqual(out.status, "permission_required")
        self.assertIn("max_capacity_gib", out.payload["unknown_fields"])
        self.assertIn("/etc/sudoers.d/dmidecode", json.dumps(out.payload))
        self.assertEqual(out.payload["data"]["readable_without_root"]["os_visible_ram_gib"], 15.25)

    def test_wsl_reports_virtualized(self):
        reg = ToolRegistry(build_source(True, "wsl2"))
        out = reg.dispatch("inspect_ram_slots_and_capacity")
        self.assertEqual(out.status, "virtualized_environment")
        self.assertTrue(out.payload["virtualized_environment"])
        self.assertIn("virtual machine", out.payload["message"])
        self.assertFalse(reg.dispatch("inspect_battery_and_power").payload["data"]["has_battery"])

    def test_minimal_uses_fallbacks(self):
        reg = ToolRegistry(build_source(True, "minimal"))
        cpu = reg.dispatch("inspect_cpu").payload["data"]
        self.assertIn("/proc/cpuinfo", cpu["method"])
        self.assertEqual(cpu["physical_cores"], 14)
        pci = reg.dispatch("inspect_pci_and_gpu").payload["data"]
        self.assertIn("/sys/bus/pci", pci["method"])
        self.assertEqual(len(pci["gpus"]), 2)
        disks = reg.dispatch("inspect_storage_devices").payload["data"]
        self.assertEqual(disks["summary"]["nvme_disks"], ["nvme0n1"])

    def test_every_tool_every_scenario_is_valid_json(self):
        for scenario in ("laptop", "laptop-nosudo", "wsl2", "minimal"):
            reg = ToolRegistry(build_source(True, scenario))
            for name in reg.names:
                out = reg.dispatch(name)
                json.loads(out.to_message_content())
                self.assertNotEqual(out.status, "error", f"{scenario}/{name}")

    def test_unknown_tool_and_junk_arguments(self):
        reg = ToolRegistry(build_source(True, "laptop"))
        self.assertEqual(reg.dispatch("run_shell", '{"cmd":"rm -rf /"}').status, "error")
        self.assertEqual(reg.dispatch("inspect_cpu", '{"cmd":"rm -rf /"}').status, "ok")
        self.assertEqual(reg.dispatch("inspect_cpu", "not json").status, "ok")

    def test_every_call_reruns_live_commands(self):
        reg = ToolRegistry(build_source(True, "laptop"))
        reg.dispatch("inspect_ram_slots_and_capacity")
        again = reg.dispatch("inspect_ram_slots_and_capacity")
        self.assertFalse(again.cached)
        self.assertTrue(any("dmidecode" in a for a in again.accessed))


# --------------------------------------------------------------------------- #
# Agent loop with a scripted fake LLM
# --------------------------------------------------------------------------- #

def _tool_msg(*names):
    calls = [SimpleNamespace(id=f"c{i}", type="function",
                             function=SimpleNamespace(name=n, arguments="{}"))
             for i, n in enumerate(names)]
    return SimpleNamespace(content=None, tool_calls=calls)


def _text_msg(text):
    return SimpleNamespace(content=text, tool_calls=None)


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=self.script.pop(0))])


def _settings(**kw):
    base = dict(provider="groq", base_url="http://x", api_key="k", model="m", mock=True)
    base.update(kw)
    return Settings(**base)


class TestAgent(unittest.TestCase):
    def test_multi_turn_tool_loop(self):
        client = FakeClient([
            _tool_msg("inspect_ram_slots_and_capacity"),
            _text_msg("Yes – one empty slot; up to 32 GB more."),
            _tool_msg("inspect_pci_and_gpu"),
            _text_msg("You have an Intel iGPU and an NVIDIA RTX 3050."),
        ])
        agent = HardwareAgent(_settings(), ToolRegistry(build_source(True, "laptop")), client)
        r1 = agent.ask("Can I add more RAM, and how much?")
        self.assertEqual([c.name for c in r1.tool_calls], ["inspect_ram_slots_and_capacity"])
        r2 = agent.ask("And what about the GPU?")
        self.assertIn("RTX 3050", r2.text)
        # second request must carry the first turn (history) + the tool result
        roles = [m["role"] for m in client.requests[-1]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant",
                                 "user", "assistant", "tool"])
        tool_payload = json.loads(client.requests[-1]["messages"][3]["content"])
        self.assertEqual(tool_payload["data"]["empty_slots"], 1)

    def test_text_json_tool_call_from_small_model(self):
        client = FakeClient([
            _text_msg('```json\n{"name": "inspect_storage_devices", "arguments": {}}\n```'),
            _text_msg("Yes, nvme0n1 is an NVMe SSD."),
        ])
        agent = HardwareAgent(_settings(provider="ollama"),
                              ToolRegistry(build_source(True, "laptop")), client)
        r = agent.ask("Is my SSD NVMe?")
        self.assertEqual([c.name for c in r.tool_calls], ["inspect_storage_devices"])
        self.assertIn("NVMe", r.text)

    def test_prose_is_not_mistaken_for_tool_call(self):
        client = FakeClient([_text_msg("Hello! Ask me about your hardware.")])
        agent = HardwareAgent(_settings(), ToolRegistry(build_source(True, "laptop")), client)
        self.assertEqual(agent.ask("hi").tool_calls, [])

    def test_round_limit_forces_final_answer(self):
        script = [_tool_msg("inspect_cpu")] * 2 + [_text_msg("Final.")]
        client = FakeClient(script)
        agent = HardwareAgent(_settings(max_tool_rounds=2),
                              ToolRegistry(build_source(True, "laptop")), client)
        r = agent.ask("cpu?")
        self.assertEqual(r.text, "Final.")
        self.assertNotIn("tools", client.requests[-1])

    def test_ollama_down_gives_actionable_error(self):
        from openai import APIConnectionError

        def refuse(**_kwargs):
            # Build the SDK's exception without depending on its HTTP library.
            exc = APIConnectionError.__new__(APIConnectionError)
            Exception.__init__(exc, "Connection error.")
            raise exc

        class Down:
            chat = SimpleNamespace(completions=SimpleNamespace(create=refuse))
        agent = HardwareAgent(_settings(provider="ollama", model="qwen2.5:3b"),
                              ToolRegistry(build_source(True, "laptop")), Down())
        r = agent.ask("cpu?")
        self.assertTrue(r.error)
        self.assertIn("ollama serve", r.text)
        self.assertEqual(agent.turns, [])  # failed turn not committed to history


if __name__ == "__main__":
    unittest.main()
