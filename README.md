# 🐧 Linux Hardware Q&A Assistant

Ask plain-English questions about your Linux laptop's hardware. The assistant answers by running **read-only, allow-listed** inspection tools on your machine. It never answers from guesswork.

```
You: Can I add more RAM, and how much?
🔧 inspect_ram_slots_and_capacity  ok   sudo -n /usr/sbin/dmidecode -t memory → ok
🤖 Yes. 1 of your 2 SODIMM slots is free. You have 16 GiB of DDR5-4800 and the firmware
   reports a 64 GiB maximum, so you can add up to 32 GiB in the empty slot (48 GiB total).
   To reach 64 GiB, use 2 × 32 GiB modules. Check your laptop's service manual first,
   because the firmware maximum and the per-slot limit are estimates.

You: And what about the GPU?
🔧 inspect_pci_and_gpu  ok   lspci -vmm -k → ok · nvidia-smi … → ok
🤖 You have hybrid graphics: an integrated Intel Iris Xe (i915) and a discrete NVIDIA
   GeForce RTX 3050 Laptop GPU with 4 GiB of VRAM on the proprietary 550.120 driver.
```

**Cost: ₹0.** It uses no paid OpenAI or Anthropic keys. It needs only three open-source packages: `openai`, `rich` and `python-dotenv`.

---

## 1. Quick start

```bash
git clone <your-repo> linux-hw-assistant && cd linux-hw-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then paste your free Groq key into .env
```

### Phase 1: develop inside WSL2 or a VM (mock hardware)

Virtual machines hide physical DIMM slots, NVMe transports and real GPUs. `--mock` replays the **exact raw output** of `lscpu`, `free`, `dmidecode`, `lsblk`, `lspci`, `nvidia-smi`, `/proc` and `/sys` from a physical laptop: an i7-12700H with 16 GB DDR5 in 1 of 2 slots, a 64 GB maximum, an NVMe SSD and Intel + NVIDIA GPUs. The real parsers, fallbacks and safety policy all run against it.

```bash
python main.py --mock                                  # chat with the simulated laptop
python main.py --mock --inspect all                    # see tool JSON (no LLM, no key needed)
python main.py --mock-scenario laptop-nosudo           # sudo needs a password → "I don't know"
python main.py --mock-scenario wsl2                    # virtualized → explicit VM refusal
python main.py --mock-scenario minimal                 # lscpu/lspci/lsblk missing → /proc,/sys fallbacks
```

| Scenario | What it simulates | What it proves |
|---|---|---|
| `laptop` (default) | Bare metal with passwordless `sudo -n dmidecode` | Full slot data and upgrade analysis |
| `laptop-nosudo` | Default Ubuntu: `sudo` asks for a password | `permission_required`, partial data, fix instructions |
| `wsl2` | Ubuntu inside WSL2 with no SMBIOS, virtual disks and no battery | `virtualized_environment: true`, no hallucinated slots |
| `minimal` | Bare metal on a minimal install | `/proc/cpuinfo`, `/sys/bus/pci`, `/sys/block` fallbacks |

### Phase 2: production on bare-metal Ubuntu

```bash
sudo apt install pciutils dmidecode util-linux     # usually already installed
python main.py                                     # real commands on this machine
python main.py --local                             # same, with a fully offline LLM
```

### CLI flags

| Flag | Purpose |
|---|---|
| `--mock` | Use the fixtures in `src/mock_data.py` instead of real commands |
| `--mock-scenario NAME` | `laptop`, `laptop-nosudo`, `wsl2` or `minimal`. Implies `--mock` |
| `--local` | Use local Ollama instead of Groq |
| `--debug` | Show the raw JSON each tool sent to the model, plus debug logs |
| `--model NAME` | Override the model name |
| `--ask "…"` | Answer one question and exit, for scripting |
| `--inspect TOOL\|all` | Print tool output only. Needs no LLM and no API key |
| `--ram-slots N` | Real number of RAM slots, for BIOSes that report phantom slots (overrides `PHYSICAL_RAM_SLOTS`) |

In-chat commands: `/tools`, `/debug`, `/reset`, `/help`, `/exit`.

---

## 2. Free LLM back-ends (same SDK, two URLs)

Both back-ends are OpenAI-API-compatible, so one `openai.OpenAI(base_url=…, api_key=…)` client covers both.

| | Groq cloud (default) | Ollama (`--local`) |
|---|---|---|
| Cost | Free tier, no credit card | Free |
| Privacy | Hardware JSON is sent to Groq | Nothing leaves your machine |
| Speed / quality | Fast, large model | Depends on your CPU/GPU; 3B models are weaker |
| Setup | Key from <https://console.groq.com/keys> | `ollama pull qwen2.5:3b` |

> **Model note (Nov 2026):** Groq **retired `llama-3.3-70b-versatile` on 16 Aug 2026** for free and developer tiers. Groq's recommended replacements are `openai/gpt-oss-120b` (the default here) or `qwen/qwen3.6-27b`. Any Groq model that supports tool calling works. Set it with `GROQ_MODEL` in `.env`.

**Ollama setup**

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b                  # or: ollama pull llama3.2:3b
OLLAMA_CONTEXT_LENGTH=8192 ollama serve # more context than the default; recommended
python main.py --local
```

Small local models sometimes print a tool call as JSON text instead of using the native `tool_calls` field. The agent detects this and recovers. It only accepts exact names of registered tools, so ordinary prose is never treated as a call.

---

## 3. Architecture

```
main.py            rich CLI: banner, spinner, "under the hood" table, Markdown answers
  └─ src/agent.py  multi-turn loop · anti-hallucination system prompt · error mapping
       └─ src/tools.py      7 tool schemas · inspectors · fallbacks · dispatcher · cache
            ├─ src/executor.py  allow-listed runner (shell=False) · path policy · VM detection
            │     ├─ LiveSource  → real subprocess / /proc / /sys
            │     └─ MockSource  → src/mock_data.py fixtures (same policy enforced)
            └─ src/parsers.py   pure text → dict parsers (unit-tested)
     src/config.py  .env + flags → immutable Settings
```

| Tool | Primary source | Fallback |
|---|---|---|
| `inspect_cpu` | `lscpu`, `lscpu -p=CPU,CORE,SOCKET,NODE` | `/proc/cpuinfo` and `/sys/devices/system/cpu/*/cpufreq` |
| `inspect_memory_usage` | `free -b` | `/proc/meminfo` |
| `inspect_ram_slots_and_capacity` | `sudo -n dmidecode -t memory` (or direct if root) | `/proc/meminfo` and `/sys/devices/virtual/dmi/id/*` |
| `inspect_storage_devices` | `lsblk -J -b -o NAME,SIZE,TYPE,MOUNTPOINT,ROTA,TRAN,MODEL`, `df` | `/sys/block/*` |
| `inspect_pci_and_gpu` | `lspci -vmm -k`, `nvidia-smi` | `/sys/bus/pci/devices/*`, amdgpu VRAM in sysfs |
| `inspect_battery_and_power` | `/sys/class/power_supply`, `/sys/class/thermal`, ACPI platform profile | – |
| `inspect_os_and_kernel` | `/etc/os-release`, `/proc`, `uname -m`, EFI and DMI | `/usr/lib/os-release`, `/proc/version` |

Every tool returns the same envelope:

```json
{"tool": "...", "status": "ok | permission_required | virtualized_environment | unavailable | tool_missing | error",
 "data_source": "live | mock:<scenario>", "virtualized_environment": false,
 "message": "why data is missing", "unknown_fields": ["..."], "data": {...}, "notes": ["..."]}
```

---

## 4. Safety

* **No arbitrary execution.** The model can only *name* one of 7 tools. Tools take no parameters that affect execution, so prompt-injected arguments are parsed, logged and ignored. Unknown tool names return an error.
* **Hard-coded allow-list.** Every possible argv is a constant tuple in `executor.ALLOWED_COMMANDS`. Requests for anything else return `not_allowed`.
* **`subprocess.run([...], shell=False, capture_output=True, text=True, timeout=5)`** with `stdin` closed, a fixed `PATH`, and `LC_ALL=C` for stable, parseable output.
* **`sudo -n` only.** It is non-interactive, so it fails instantly instead of prompting. It is used for exactly one command: `dmidecode -t memory`.
* **File-read policy.** Reads are allowed only from specific `/proc` files and `/sys` subtrees. `..` traversal and symlinks that resolve outside the allowed roots are rejected. Paths containing `serial`, `uuid`, `asset_tag`, `environ`, `kcore`, PCI `config` and similar are always denied.
* **Tool output is data.** The system prompt tells the model to ignore instructions embedded in tool output. Hardware strings such as a USB stick's model name are attacker-controllable.
* **Fails safe.** A crashing inspector returns `status: "error"`; it never takes down the chat.

## 5. Permissions (RAM slots need root)

`dmidecode` reads SMBIOS tables that only root can read. Without root, the assistant still reports OS-visible RAM and the system vendor and model. For slot count, free slots and maximum capacity, it returns `permission_required` with instructions, and the model says *"I don't know"*.

There are two ways to fix this. The second is recommended.

```bash
# 1) Run the whole app as root (simple, but broader privilege than needed)
sudo .venv/bin/python main.py   # use the venv python: sudo resets PATH

# 2) Allow only this read-only command without a password
echo "$USER ALL=(root) NOPASSWD: /usr/sbin/dmidecode -t memory" | sudo tee /etc/sudoers.d/dmidecode
sudo chmod 440 /etc/sudoers.d/dmidecode && sudo visudo -c
```

The rule pins both the binary path and the arguments, so it cannot be used to run `dmidecode` with other options. The app resolves `dmidecode`'s absolute path at run time so the call matches the rule.

## 6. Accuracy and anti-hallucination

* **Structured JSON in, not raw text.** Parsers normalise units (GiB vs decimal GB), drop placeholders such as "Not Specified" and "To be filled by O.E.M.", and flag `unknown_fields`.
* **Arithmetic is done in Python, not by the LLM.** `upgrade_analysis` precomputes headroom, the per-slot estimate, the most you can add without replacing modules, and the module type to match.
* **Caveats are explicit.** The SMBIOS *Maximum Capacity* often reflects the CPU's memory-controller limit, not the laptop maker's validated limit. The per-slot maximum is a heuristic (max ÷ socketed slots). Soldered memory (`Row Of Chips` or LPDDR) is detected and marked as not replaceable.
* **Hybrid CPUs.** On Intel 12th gen and later, plain `lscpu` reports "2 threads per core" even though E-cores have one. The assistant uses `lscpu -p` topology for exact core and thread counts.
* **Installed vs visible RAM.** 16 GB installed shows as about 15.25 GiB in `free`, because firmware and the iGPU reserve some. The two figures are kept separate.
* **Virtualization.** The assistant checks `systemd-detect-virt`, `/proc/version` (WSL), the CPU `hypervisor` flag and the DMI product name. In a VM, RAM-slot answers return `virtualized_environment` and the model must say *"I cannot read physical RAM slots because this is running inside a virtual machine"*. Disk, GPU and CPU answers carry a note that they describe virtual devices.
* **System prompt rules.** The model may only state facts that come from tool results, must say "I don't know" for anything unknown, and must label general knowledge as general.

> An LLM can still misphrase things, especially 3B local models. Use `--debug` to see exactly what data the model received and check the answer against it.

## 7. Portability

* Tested on x86-64 Ubuntu. The parsers also handle older util-linux and procps formats: flat vs tree `lscpu`, string vs boolean `lsblk`, legacy `free` columns, and ARM `/proc/cpuinfo`.
* Every binary is checked with `shutil.which()` before it runs. Missing tools degrade to `/proc` and `/sys`. The PCI fallback shows vendor and class but not marketing names, because those come from `pci.ids` via `lspci`.
* Battery data handles both `energy_*` (µWh) and `charge_*` (µAh) reporting. Peripheral batteries such as a wireless mouse are skipped.
* Requires Python 3.10+.

## 8. Privacy

| Data | Handling |
|---|---|
| DIMM serial numbers and asset tags | Dropped by the parser and never sent |
| System serial and UUID (`/sys/.../product_serial`, `product_uuid`) | Blocked by the path policy and never read |
| Username in mount paths (`/home/alex`, `/media/alex/…`) | Redacted to `/<user>` |
| Hostname, MAC addresses, IP addresses | Never collected |
| Hardware specs (CPU, RAM, disks, GPU, battery, OS) | Sent to the LLM. With Groq they leave your machine; with `--local` they never do |

If the hardware inventory itself is sensitive (for example on corporate machines), use `--local`.

## 9. Testing

```bash
python -m unittest discover -s tests -t . -v
```

The tests cover:

* every parser, including legacy formats, soldered RAM and both battery styles
* the security policy: blocked commands, path traversal, serial denial, `shell=False` and timeout
* all 7 tools under all 4 scenarios
* the agent loop against a scripted fake LLM: multi-turn history, JSON-text tool calls, the round limit, and an actionable error when Ollama is down

No network access or API key is needed.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `GROQ_API_KEY is not set` | Get a free key at console.groq.com/keys and put it in `.env`, or use `--local` |
| `Groq does not offer model …` | The model was retired. Set `GROQ_MODEL=openai/gpt-oss-120b` |
| `Groq free-tier rate limit reached` | Wait a minute, lower `HISTORY_CHAR_BUDGET_CLOUD`, or use `--local` |
| `Cannot reach Ollama` | Run `ollama serve`, then `ollama pull qwen2.5:3b` |
| Local model ignores tools or gives odd answers | Use `qwen2.5:3b` or `llama3.2:3b`, start Ollama with `OLLAMA_CONTEXT_LENGTH=8192`, and `/reset` long chats |
| RAM slots say `permission_required` | See [Permissions](#5-permissions-ram-slots-need-root) |
| GPU name shows `device 10de:25a2` | Install `pciutils` (`lspci`) |
| No VRAM for NVIDIA | Install the proprietary driver, which provides `nvidia-smi` |

## 11. Extending

To add a tool:

1. Add any new command to `ALLOWED_COMMANDS` (and any new `/sys` prefix to `ALLOWED_PREFIXES`).
2. Write a pure parser in `parsers.py` and test it with a fixture in `mock_data.py`.
3. Write an inspector in `tools.py` that returns `_envelope(...)`.
4. Register a `ToolSpec` with a clear description. The description is how the model decides when to call the tool.
