"""Safe, read-only system access layer.

Security model
--------------
* The LLM NEVER supplies a command line. It can only name a *tool*; tools ask
  this module for a command by its **allow-list key** (e.g. ``"lscpu"``). The
  argv for every key is a hard-coded tuple in :data:`ALLOWED_COMMANDS`.
* Every command runs with ``subprocess.run([...], shell=False, ...)`` – no
  shell, no string interpolation, no globbing, ``stdin`` closed, a fixed
  ``PATH`` and ``LC_ALL=C`` (stable, parseable output), and a hard timeout.
* ``sudo`` is only ever used as ``sudo -n`` (non-interactive: it fails
  instantly instead of prompting) and only for ``dmidecode -t memory``.
* File reads are restricted to an allow-list of ``/proc`` and ``/sys`` paths,
  with identifiers such as serial numbers and UUIDs explicitly denied.

Two interchangeable back-ends implement :class:`SystemSource`:

* :class:`LiveSource` – the real machine.
* :class:`MockSource` – fixture data from :mod:`src.mock_data`. It enforces
  the *same* allow-list, so mock mode exercises the real policy code.
"""

from __future__ import annotations

import logging
import os
import posixpath
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Allow-lists
# --------------------------------------------------------------------------- #

#: Every command the application can ever run. Keys are what tools request.
#: ``{dmidecode}`` is replaced by the absolute path of dmidecode at run time so
#: that a sudoers rule can pin the exact binary.
ALLOWED_COMMANDS: dict[str, tuple[str, ...]] = {
    "lscpu": ("lscpu",),
    "lscpu_topology": ("lscpu", "-p=CPU,CORE,SOCKET,NODE"),
    "free": ("free", "-b"),
    "dmidecode_memory_sudo": ("sudo", "-n", "{dmidecode}", "-t", "memory"),
    "dmidecode_memory_root": ("{dmidecode}", "-t", "memory"),
    "lsblk": ("lsblk", "-J", "-b", "-o", "NAME,SIZE,TYPE,MOUNTPOINT,ROTA,TRAN,MODEL"),
    "df": (
        "df", "-B1", "-x", "tmpfs", "-x", "devtmpfs", "-x", "squashfs", "-x", "overlay",
        "-x", "efivarfs", "--output=source,fstype,size,used,avail,target",
    ),
    "lspci": ("lspci", "-vmm", "-k"),
    "nvidia_smi": (
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,temperature.gpu,power.draw,utilization.gpu",
        "--format=csv,noheader,nounits",
    ),
    "systemd_detect_virt": ("systemd-detect-virt",),
    "uname_machine": ("uname", "-m"),
}

#: Return codes that are *not* failures for specific commands.
#: ``systemd-detect-virt`` prints "none" and exits 1 on bare metal.
OK_RETURNCODES: dict[str, frozenset[int]] = {
    "systemd_detect_virt": frozenset({0, 1}),
}

#: Exact files that may be read.
ALLOWED_EXACT_FILES: frozenset[str] = frozenset({
    "/proc/cpuinfo",
    "/proc/meminfo",
    "/proc/version",
    "/proc/uptime",
    "/proc/swaps",
    "/proc/sys/kernel/osrelease",
    "/proc/sys/kernel/ostype",
    "/etc/os-release",
    "/usr/lib/os-release",
    "/sys/firmware/acpi/platform_profile",
    "/sys/firmware/acpi/platform_profile_choices",
    "/sys/firmware/efi/fw_platform_size",
})

#: Directory prefixes whose contents may be read / listed.
ALLOWED_PREFIXES: tuple[str, ...] = (
    "/sys/devices/virtual/dmi/id/",
    "/sys/class/power_supply/",
    "/sys/devices/system/cpu/",
    "/sys/bus/pci/devices/",
    "/sys/block/",
    "/sys/class/drm/",
    "/sys/class/thermal/",
    "/sys/firmware/efi/",
)

#: Path fragments that are never read even inside an allowed prefix
#: (unique identifiers, raw memory, binary config space, process data).
DENIED_FRAGMENTS: tuple[str, ...] = (
    "serial", "uuid", "asset_tag", "/environ", "/dev/mem", "kcore", "/config",
    "/rom", "/resource", "/efivars", "/vars/",
)

#: Real (symlink-resolved) locations a read may end up in.
ALLOWED_REAL_ROOTS: tuple[str, ...] = ("/proc/", "/sys/", "/etc/", "/usr/lib/")

SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
MAX_READ_BYTES = 256 * 1024

PERMISSION_MARKERS: tuple[str, ...] = (
    "a password is required",
    "a terminal is required",
    "permission denied",
    "not in the sudoers",
    "may not run sudo",
    "is not allowed to execute",
    "operation not permitted",
    "must be run as root",
    "you must be root",
    "no tty present",
)
NO_SMBIOS_MARKERS: tuple[str, ...] = (
    "no smbios nor dmi entry point found",
    "no smbios entry point",
)


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass
class CommandResult:
    """Outcome of one allow-listed command."""

    key: str
    argv: list[str]
    ok: bool
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    #: None | missing_binary | permission_denied | no_smbios | timeout |
    #: nonzero_exit | os_error | not_allowed
    error_kind: str | None = None
    duration_ms: int = 0

    @property
    def display(self) -> str:
        return " ".join(self.argv) if self.argv else self.key

    def summary(self) -> str:
        status = "ok" if self.ok else (self.error_kind or "failed")
        return f"{self.display} → {status}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FileResult:
    path: str
    ok: bool
    content: str = ""
    error_kind: str | None = None  # not_allowed | not_found | permission_denied | os_error


@dataclass
class AccessLog:
    """Records what a tool touched, for the 'under the hood' display."""

    entries: list[str] = field(default_factory=list)

    def add(self, text: str) -> None:
        self.entries.append(text)


class PolicyViolation(PermissionError):
    """Raised when code (never the LLM directly) requests a non-allowed action."""


# --------------------------------------------------------------------------- #
# Policy helpers (shared by live + mock)
# --------------------------------------------------------------------------- #

def classify_failure(stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}".lower()
    if any(m in text for m in NO_SMBIOS_MARKERS):
        return "no_smbios"
    if any(m in text for m in PERMISSION_MARKERS):
        return "permission_denied"
    return "nonzero_exit"


def normalize_path(path: str) -> str:
    """Collapse ``..`` / ``//`` lexically so traversal tricks are rejected."""
    if not path.startswith("/"):
        raise PolicyViolation(f"Relative paths are not allowed: {path!r}")
    return posixpath.normpath(path)


def is_path_allowed(path: str) -> bool:
    try:
        norm = normalize_path(path)
    except PolicyViolation:
        return False
    lowered = norm.lower()
    if any(frag in lowered for frag in DENIED_FRAGMENTS):
        return False
    if norm in ALLOWED_EXACT_FILES:
        return True
    probe = norm + "/"
    return any(probe.startswith(prefix) or norm.startswith(prefix) for prefix in ALLOWED_PREFIXES)


# --------------------------------------------------------------------------- #
# Abstract source
# --------------------------------------------------------------------------- #

class SystemSource(ABC):
    """Uniform interface over a live machine or mock fixtures."""

    name: str = "abstract"

    def __init__(self, timeout: int = 5) -> None:
        self.timeout = timeout

    # ---- primitives each backend implements ----------------------------- #
    @abstractmethod
    def which(self, binary: str) -> str | None: ...

    @abstractmethod
    def is_root(self) -> bool: ...

    @abstractmethod
    def _execute(self, key: str, argv: list[str]) -> CommandResult: ...

    @abstractmethod
    def _read(self, path: str) -> FileResult: ...

    @abstractmethod
    def _listdir(self, path: str) -> list[str] | None: ...

    @abstractmethod
    def _readlink(self, path: str) -> str | None: ...

    @abstractmethod
    def _exists(self, path: str) -> bool: ...

    # ---- policy-enforcing public API ------------------------------------ #
    def resolve_argv(self, key: str) -> list[str] | CommandResult:
        """Turn an allow-list key into a concrete argv, or a failure result."""
        template = ALLOWED_COMMANDS.get(key)
        if template is None:
            log.warning("Blocked non-allow-listed command key %r", key)
            return CommandResult(key=key, argv=[], ok=False, error_kind="not_allowed",
                                 stderr=f"Command key {key!r} is not allow-listed.")
        argv: list[str] = []
        for part in template:
            if part == "{dmidecode}":
                resolved = self.which("dmidecode")
                if not resolved:
                    return CommandResult(key=key, argv=["dmidecode", "-t", "memory"], ok=False,
                                         error_kind="missing_binary",
                                         stderr="dmidecode is not installed.")
                argv.append(resolved)
            else:
                argv.append(part)
        if not self.which(argv[0]):
            return CommandResult(key=key, argv=argv, ok=False, error_kind="missing_binary",
                                 stderr=f"{argv[0]} is not installed or not on PATH.")
        return argv

    def run(self, key: str) -> CommandResult:
        resolved = self.resolve_argv(key)
        if isinstance(resolved, CommandResult):
            return resolved
        result = self._execute(key, resolved)
        if not result.ok and result.error_kind is None:
            result.error_kind = classify_failure(result.stdout, result.stderr)
        log.debug("cmd %s", result.summary())
        return result

    def read_text(self, path: str) -> FileResult:
        if not is_path_allowed(path):
            log.warning("Blocked read of non-allow-listed path %r", path)
            return FileResult(path=path, ok=False, error_kind="not_allowed")
        return self._read(normalize_path(path))

    def read_value(self, path: str) -> str | None:
        """Read a small sysfs/proc value, stripped; ``None`` if unavailable."""
        res = self.read_text(path)
        return res.content.strip() if res.ok else None

    def list_dir(self, path: str) -> list[str]:
        if not is_path_allowed(path):
            return []
        return sorted(self._listdir(normalize_path(path)) or [])

    def read_link(self, path: str) -> str | None:
        if not is_path_allowed(path):
            return None
        return self._readlink(normalize_path(path))

    def exists(self, path: str) -> bool:
        if not is_path_allowed(path):
            return False
        return self._exists(normalize_path(path))


# --------------------------------------------------------------------------- #
# Live backend
# --------------------------------------------------------------------------- #

class LiveSource(SystemSource):
    name = "live"

    def which(self, binary: str) -> str | None:
        if os.path.isabs(binary):
            return binary if os.access(binary, os.X_OK) else None
        return shutil.which(binary, path=SAFE_PATH)

    def is_root(self) -> bool:
        return hasattr(os, "geteuid") and os.geteuid() == 0

    def _execute(self, key: str, argv: list[str]) -> CommandResult:
        env = {"PATH": SAFE_PATH, "LC_ALL": "C", "LANG": "C"}
        started = time.monotonic()
        try:
            proc = subprocess.run(
                argv,                      # a list – never a shell string
                shell=False,               # never a shell
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                env=env,
                stdin=subprocess.DEVNULL,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(key=key, argv=argv, ok=False, error_kind="timeout",
                                 stderr=f"Timed out after {exc.timeout}s",
                                 duration_ms=int((time.monotonic() - started) * 1000))
        except FileNotFoundError:
            return CommandResult(key=key, argv=argv, ok=False, error_kind="missing_binary",
                                 stderr=f"{argv[0]} not found")
        except PermissionError as exc:
            return CommandResult(key=key, argv=argv, ok=False,
                                 error_kind="permission_denied", stderr=str(exc))
        except OSError as exc:
            return CommandResult(key=key, argv=argv, ok=False, error_kind="os_error",
                                 stderr=str(exc))

        elapsed = int((time.monotonic() - started) * 1000)
        ok_codes = OK_RETURNCODES.get(key, frozenset({0}))
        ok = proc.returncode in ok_codes
        result = CommandResult(key=key, argv=argv, ok=ok, returncode=proc.returncode,
                               stdout=proc.stdout or "", stderr=proc.stderr or "",
                               duration_ms=elapsed)
        # dmidecode can exit 0 yet report "No SMBIOS nor DMI entry point found".
        if ok and any(m in result.stdout.lower() for m in NO_SMBIOS_MARKERS):
            result.ok = False
            result.error_kind = "no_smbios"
        return result

    def _read(self, path: str) -> FileResult:
        real = os.path.realpath(path)
        if not any((real + "/").startswith(root) or real.startswith(root)
                   for root in ALLOWED_REAL_ROOTS):
            return FileResult(path=path, ok=False, error_kind="not_allowed")
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return FileResult(path=path, ok=True, content=fh.read(MAX_READ_BYTES))
        except FileNotFoundError:
            return FileResult(path=path, ok=False, error_kind="not_found")
        except PermissionError:
            return FileResult(path=path, ok=False, error_kind="permission_denied")
        except OSError:
            # Some sysfs attributes raise EIO/ENODEV/EINVAL when unsupported.
            return FileResult(path=path, ok=False, error_kind="os_error")

    def _listdir(self, path: str) -> list[str] | None:
        try:
            return os.listdir(path)
        except OSError:
            return None

    def _readlink(self, path: str) -> str | None:
        try:
            return os.readlink(path)
        except OSError:
            return None

    def _exists(self, path: str) -> bool:
        return os.path.exists(path)


# --------------------------------------------------------------------------- #
# Mock backend
# --------------------------------------------------------------------------- #

class MockSource(SystemSource):
    """Serves fixtures from :mod:`src.mock_data`, enforcing the same policy."""

    name = "mock"

    def __init__(self, fixture: dict[str, Any], timeout: int = 5) -> None:
        super().__init__(timeout=timeout)
        self.fixture = fixture
        self.scenario = fixture.get("scenario", "custom")
        self._files: dict[str, str] = fixture.get("files", {})
        self._links: dict[str, str] = fixture.get("links", {})
        self._commands: dict[str, dict[str, Any]] = fixture.get("commands", {})
        self._binaries: dict[str, str] = fixture.get("binaries", {})
        self._root: bool = bool(fixture.get("root", False))

    def which(self, binary: str) -> str | None:
        if os.path.isabs(binary):
            return binary if binary in self._binaries.values() else None
        return self._binaries.get(binary)

    def is_root(self) -> bool:
        return self._root

    def _execute(self, key: str, argv: list[str]) -> CommandResult:
        spec = self._commands.get(key)
        if spec is None:
            return CommandResult(key=key, argv=argv, ok=False, returncode=1,
                                 stderr="(mock) no fixture for this command",
                                 error_kind="nonzero_exit")
        rc = int(spec.get("rc", 0))
        ok_codes = OK_RETURNCODES.get(key, frozenset({0}))
        result = CommandResult(key=key, argv=argv, ok=rc in ok_codes, returncode=rc,
                               stdout=spec.get("stdout", ""), stderr=spec.get("stderr", ""),
                               duration_ms=int(spec.get("duration_ms", 3)))
        if spec.get("timeout"):
            result.ok, result.error_kind = False, "timeout"
        elif result.ok and any(m in result.stdout.lower() for m in NO_SMBIOS_MARKERS):
            result.ok, result.error_kind = False, "no_smbios"
        return result

    def _read(self, path: str) -> FileResult:
        if path in self._files:
            return FileResult(path=path, ok=True, content=self._files[path])
        return FileResult(path=path, ok=False, error_kind="not_found")

    def _listdir(self, path: str) -> list[str] | None:
        prefix = path.rstrip("/") + "/"
        children = {
            p[len(prefix):].split("/", 1)[0]
            for p in list(self._files) + list(self._links)
            if p.startswith(prefix)
        }
        return sorted(children) if children else None

    def _readlink(self, path: str) -> str | None:
        return self._links.get(path)

    def _exists(self, path: str) -> bool:
        if path in self._files or path in self._links:
            return True
        prefix = path.rstrip("/") + "/"
        return any(p.startswith(prefix) for p in list(self._files) + list(self._links))


# --------------------------------------------------------------------------- #
# Environment checks
# --------------------------------------------------------------------------- #

_VM_PRODUCT_HINTS = (
    "virtualbox", "vmware", "kvm", "qemu", "virtual machine", "hyper-v", "bochs",
    "xen", "parallels", "standard pc (",
)


def detect_virtualization(source: SystemSource, log_: AccessLog | None = None) -> dict[str, Any]:
    """Best-effort VM / container / WSL detection.

    Order: ``systemd-detect-virt`` → ``/proc/version`` (WSL) →
    ``hypervisor`` CPU flag → DMI product name.
    """
    evidence: list[str] = []
    vtype: str | None = None

    res = source.run("systemd_detect_virt")
    if log_:
        log_.add(res.summary())
    if res.ok:
        out = res.stdout.strip().lower()
        if out and out != "none":
            vtype = out
            evidence.append(f"systemd-detect-virt reported '{out}'")
        elif out == "none":
            evidence.append("systemd-detect-virt reported 'none'")

    version = source.read_value("/proc/version") or ""
    if "microsoft" in version.lower() or "wsl" in version.lower():
        vtype = vtype or "wsl"
        evidence.append("/proc/version mentions Microsoft/WSL")

    cpuinfo = source.read_text("/proc/cpuinfo")
    if cpuinfo.ok:
        for line in cpuinfo.content.splitlines():
            if line.lower().startswith("flags") and " hypervisor" in f" {line.split(':', 1)[-1]} ":
                vtype = vtype or "unknown-hypervisor"
                evidence.append("CPU 'hypervisor' flag is set in /proc/cpuinfo")
                break

    product = (source.read_value("/sys/devices/virtual/dmi/id/product_name") or "").lower()
    sys_vendor = (source.read_value("/sys/devices/virtual/dmi/id/sys_vendor") or "").lower()
    if any(h in product or h in sys_vendor for h in _VM_PRODUCT_HINTS):
        vtype = vtype or "vm"
        evidence.append(f"DMI product/vendor looks virtual ('{product or sys_vendor}')")

    containers = {"docker", "podman", "lxc", "lxc-libvirt", "systemd-nspawn", "rkt",
                  "openvz", "container-other", "proot"}
    return {
        "virtualized": vtype is not None,
        "type": vtype,
        "is_container": vtype in containers if vtype else False,
        "evidence": evidence,
    }


def build_source(mock: bool, scenario: str = "laptop", timeout: int = 5) -> SystemSource:
    if mock:
        from .mock_data import get_fixture  # local import keeps live mode lean
        return MockSource(get_fixture(scenario), timeout=timeout)
    return LiveSource(timeout=timeout)
