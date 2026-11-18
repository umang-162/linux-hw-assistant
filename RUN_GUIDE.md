# Run Guide: Linux Hardware Q&A Assistant

This guide covers three stages, in order:

| Stage | Where | Mode | Goal |
|---|---|---|---|
| **A** | Your **Windows laptop**, inside WSL2 (Ubuntu) | `--mock` | Develop and test with simulated laptop hardware |
| **B** | Back to **normal Windows** | – | Close WSL, find your files, pack the project |
| **C** | The **other PC running Ubuntu** directly | normal (live) | Real answers about real hardware |

> **Why two machines?** WSL2 is a virtual machine. It cannot see your laptop's real RAM slots, battery or GPU details, so `--mock` replays output from a real laptop instead. On a PC where Ubuntu is installed directly ("bare metal"), the assistant runs the real commands.

---

## Part A: Run on Windows with WSL2 (mock mode)

### A1. Install WSL2 and Ubuntu (one time)

1. Right-click **Start** and choose **Terminal (Admin)** or **Windows PowerShell (Admin)**.
2. Run:
   ```powershell
   wsl --install -d Ubuntu
   ```
3. **Restart** the laptop when asked.
4. Ubuntu opens automatically after the restart. If it doesn't, search **Ubuntu** in the Start menu.
5. Choose a Linux **username** and **password**. You'll need this password for `sudo`. Nothing appears on screen while you type it; that's normal.

Check that it worked by running this in PowerShell (it should show `Ubuntu` with `VERSION 2`):
```powershell
wsl -l -v
```

### A2. Install system packages (one time)

In the **Ubuntu** window:
```bash
sudo apt update
sudo apt install -y python3-venv python3-pip unzip zip git pciutils dmidecode
```

### A3. Copy the project into Ubuntu

Your Windows drives appear inside Ubuntu under `/mnt/`. For example, `C:\Users\Umang\Downloads` is `/mnt/c/Users/Umang/Downloads`.

```bash
ls /mnt/c/Users/                        # find your Windows user folder name
cp /mnt/c/Users/<YourWindowsName>/Downloads/linux-hw-assistant.zip ~
cd ~
unzip linux-hw-assistant.zip
cd linux-hw-assistant
```

> **Tip:** keep the project in your Ubuntu home folder (`~`), not under `/mnt/c/`. Files on the Windows drive are much slower to use from WSL.

### A4. Create the Python environment (one time)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Your prompt now starts with `(.venv)`, which means the environment is active.

### A5. Test the tools (no API key needed)

```bash
python main.py --mock --inspect all
```
You should see 7 JSON panels: CPU, memory usage, RAM slots, storage, PCI/GPU, battery and OS.

To run the automated tests:
```bash
python -m unittest discover -s tests -t . -v
```
The last line should say `OK`.

### A6. Add your free Groq API key

1. Open https://console.groq.com/keys, sign in, click **Create API Key** and copy it. It's free and needs no card.
2. Create and edit the settings file:
   ```bash
   cp .env.example .env
   nano .env
   ```
3. Replace `gsk_your_free_groq_key_here` with your key.
4. Save with **Ctrl+O**, press **Enter**, then exit with **Ctrl+X**.

> `.env` is listed in `.gitignore`, so your key is never uploaded to GitHub.

### A7. Chat with the simulated laptop

```bash
python main.py --mock
```

Try these questions in order (the 2nd and 3rd test follow-ups):
```
Can I add more RAM, and how much?
And what about the GPU?
Is my SSD NVMe?
How healthy is my battery?
How many cores and threads does my CPU have?
```

Commands you can type in the chat:

| Command | Effect |
|---|---|
| `/tools` | List the inspection tools |
| `/debug` | Show the raw JSON each tool sent to the model |
| `/reset` | Clear the conversation |
| `/exit` | Quit |

### A8. Test the edge cases

```bash
python main.py --mock-scenario laptop-nosudo   # sudo needs a password → it must say "I don't know" for slots
python main.py --mock-scenario wsl2           # inside a VM → it must refuse to guess RAM slots
python main.py --mock-scenario minimal        # lscpu/lspci/lsblk missing → /proc and /sys fallbacks
python main.py --mock --debug                 # see exactly what data the model receives
python main.py --mock --ask "Is my SSD NVMe?"  # one question, then exit
```

### A9 (optional). Fully offline mode with Ollama

Install Ollama **inside Ubuntu**, not the Windows app. WSL can't reach a Windows Ollama at `localhost` by default.
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b
python main.py --mock --local
```
If you see "Cannot reach Ollama", open a **second** Ubuntu window and leave this running:
```bash
OLLAMA_CONTEXT_LENGTH=8192 ollama serve
```

### A10. What happens without `--mock` inside WSL

```bash
python main.py --inspect all
```
This runs the real commands, but WSL is a VM. Expect `virtualized_environment: true`, no RAM slot data and no battery. **That is correct behaviour**: the assistant reports what it can't see instead of guessing.

---

## Part B: Switch back to normal Windows

### B1. Leave the app and Ubuntu

```bash
/exit           # inside the chat
deactivate      # leave the Python environment
exit            # close the Ubuntu terminal
```

Ubuntu keeps running quietly in the background for a few minutes. To stop it at once and free its RAM, run this in **PowerShell**:
```powershell
wsl --shutdown
```

### B2. Open your project from Windows File Explorer

Type this into the File Explorer address bar:
```
\\wsl$\Ubuntu\home\<your-linux-username>\linux-hw-assistant
```
You can also click **Linux → Ubuntu** in the File Explorer sidebar.

To edit the code in VS Code, run this from the project folder in Ubuntu (VS Code sets up WSL support automatically):
```bash
code .
```

### B3. Come back later

Open **Ubuntu** from the Start menu, then:
```bash
cd ~/linux-hw-assistant
source .venv/bin/activate
python main.py --mock
```

### B4. Pack the project for the Ubuntu PC

Choose **one** option.

**Option 1: GitHub (recommended)**

1. Create an **empty** repository on GitHub, for example `linux-hw-assistant`. Don't add a README.
2. In Ubuntu, run:
   ```bash
   cd ~/linux-hw-assistant
   git init
   git add .
   git commit -m "Linux Hardware Q&A Assistant"
   git branch -M main
   git remote add origin https://github.com/<your-username>/linux-hw-assistant.git
   git push -u origin main
   ```
3. When git asks for a password, paste a **GitHub Personal Access Token**, not your account password. Create one under GitHub → Settings → Developer settings → Personal access tokens.

**Option 2: zip file (USB drive / Google Drive / email)**

```bash
cd ~
zip -r linux-hw-assistant.zip linux-hw-assistant \
    -x 'linux-hw-assistant/.venv/*' 'linux-hw-assistant/.env' '*/__pycache__/*'
cp linux-hw-assistant.zip /mnt/c/Users/<YourWindowsName>/Desktop/
```
The zip is now on your Windows Desktop. The `.venv` folder and your `.env` key are left out on purpose: the environment is rebuilt on the new PC, and the key stays private.

---

## Part C: Run on the Ubuntu PC (real hardware)

Do everything below in a **Terminal** on the Ubuntu PC (press **Ctrl+Alt+T**).

### C1. Confirm it's bare metal

```bash
systemd-detect-virt
```
- `none` means bare metal, and you'll get real hardware answers.
- Anything else (`kvm`, `oracle`, `vmware`, `wsl`) means it's a VM, and slot and GPU data will be hidden.

### C2. Install system packages (one time)

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git unzip pciutils dmidecode util-linux procps
```
Optional, for NVIDIA GPU VRAM and temperature: install the proprietary driver from **Software & Updates → Additional Drivers**. It provides `nvidia-smi`.

### C3. Get the project

From GitHub:
```bash
cd ~
git clone https://github.com/<your-username>/linux-hw-assistant.git
cd linux-hw-assistant
```

Or from the zip, after copying it to `~/Downloads` from a USB drive or Drive:
```bash
cd ~
unzip ~/Downloads/linux-hw-assistant.zip
cd linux-hw-assistant
```

### C4. Create the environment and add your key

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env        # paste your GROQ_API_KEY, then Ctrl+O, Enter, Ctrl+X
```

### C5. First real run (no API key needed)

```bash
python main.py --inspect all
```
Check these results:

| Tool | Expected on bare metal |
|---|---|
| any tool | `"virtualized_environment": false` |
| `inspect_cpu` | your real CPU model, cores and threads |
| `inspect_storage_devices` | your real disk; `is_nvme: true` if it's NVMe |
| `inspect_pci_and_gpu` | your real GPU(s) |
| `inspect_ram_slots_and_capacity` | **`permission_required`** for now, which is fixed in the next step |

### C6. Allow RAM-slot reading (needs root)

`dmidecode` must run as root to read RAM slot data. Choose **one** option.

**Option 1: recommended.** Allow only this one read-only command without a password:
```bash
echo "$USER ALL=(root) NOPASSWD: $(command -v dmidecode) -t memory" | sudo tee /etc/sudoers.d/dmidecode
sudo chmod 440 /etc/sudoers.d/dmidecode
sudo visudo -c
```
`visudo -c` must report the files as **parsed OK**. If it shows an error, delete the file with `sudo rm /etc/sudoers.d/dmidecode` and try again.

Test it. This should print memory information without asking for a password:
```bash
sudo -n dmidecode -t memory | head -20
```

**Option 2:** run the whole app as root. This is simpler but gives the app more power than it needs:
```bash
sudo .venv/bin/python main.py
```
Use the full `.venv/bin/python` path, because `sudo` resets `PATH` and plain `python` would miss the installed packages.

Then check again:
```bash
python main.py --inspect inspect_ram_slots_and_capacity
```
`status` should now be **`ok`**, with `total_slots`, `populated_slots`, `max_capacity_gib` and an `upgrade_analysis`.

### C7. Chat about the real machine

```bash
python main.py            # Groq cloud (fast; hardware details are sent to Groq)
python main.py --local    # Ollama (offline; nothing leaves the PC) – see C8
```
This is **normal mode**: just leave out `--mock`.

### C8 (optional). Fully offline on the Ubuntu PC

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b
sudo systemctl edit ollama
```
In the editor that opens, add these lines, then save and exit:
```
[Service]
Environment="OLLAMA_CONTEXT_LENGTH=8192"
```
Then restart Ollama and run the app:
```bash
sudo systemctl restart ollama
python main.py --local
```

### C9. Daily use

```bash
cd ~/linux-hw-assistant
source .venv/bin/activate
python main.py
```

### C10. Updating after changes made on Windows

If you used GitHub, push from WSL:
```bash
git add . && git commit -m "update" && git push
```
Then pull on the Ubuntu PC:
```bash
cd ~/linux-hw-assistant && git pull
source .venv/bin/activate && pip install -r requirements.txt
```

---

## Mode cheat-sheet

| You want | Command |
|---|---|
| Simulated laptop, Groq | `python main.py --mock` |
| Simulated laptop, offline | `python main.py --mock --local` |
| **Real hardware (normal mode)**, Groq | `python main.py` |
| Real hardware, offline | `python main.py --local` |
| Tool output only, no AI or key | `python main.py --inspect all` (add `--mock` for simulated) |
| One question and exit | `python main.py --ask "How much RAM do I have?"` |
| See the data the AI receives | add `--debug` |
| Use a different Groq model | add `--model openai/gpt-oss-20b`, or set `GROQ_MODEL` in `.env` |

## Troubleshooting

| Problem | Fix |
|---|---|
| `wsl --install` says WSL is already installed | Run `wsl --install -d Ubuntu`, or `wsl --update` |
| `python3 -m venv` fails with "ensurepip is not available" | `sudo apt install -y python3-venv` |
| `ModuleNotFoundError: rich` / `openai` | Activate the environment first: `source .venv/bin/activate` |
| `GROQ_API_KEY is not set` | Fill in `.env` (A6 / C4), or use `--local` |
| `Groq does not offer model …` | Set `GROQ_MODEL=openai/gpt-oss-120b` in `.env` |
| `rate limit reached` | Wait a minute, or use `--local` |
| `Cannot reach Ollama` | Run `ollama serve` in another terminal, then `ollama pull qwen2.5:3b` |
| RAM slots say `permission_required` | Do step C6 |
| RAM slots say `virtualized_environment` | You're in WSL or a VM. Use `--mock`, or run on the bare-metal Ubuntu PC |
| GPU shows as `device 10de:xxxx` | `sudo apt install pciutils` |
| NVIDIA VRAM unknown | Install the proprietary NVIDIA driver, which provides `nvidia-smi` |
| `git push` rejects your password | Use a Personal Access Token instead of your password |
