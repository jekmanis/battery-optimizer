#!/usr/bin/env python
"""
Build, copy and (re)install the battery optimizer add-on on the HA machine.

Replaces deploy.ps1's role. A local add-on is a directory under the HA
``addons`` share; the Supervisor builds its image from that directory. So a
deploy is:

  1. preflight   - git state (+ commit it deploys), pytest, py_compile;
  2. stage       - addon/battery_optimizer/ + the optimizer code in app/,
                   config.yaml ``version`` stamped with APP_VERSION;
  3. backup      - the current share copy -> <share>/share/battery_optimizer_backups/
                   (NEVER under addons/: the Supervisor scans that tree for
                   config.yaml, and a backup there would be a second add-on
                   with the same slug - the apps/ shadowing incident again);
  4. copy        - into \\\\<ha>\\addons\\battery_optimizer, pruning files the
                   stage no longer has, then SHA256-verify every file;
  5. supervisor  - store reload, then install / update / rebuild, optional
                   options, start - through HA's websocket ``supervisor/api``
                   with the ADMIN token (the REST /api/hassio/addons/<slug>/info
                   path answers 401 for every token on this installation);
  6. verify      - read the add-on log for the version line and for
                   Traceback / ModuleNotFoundError / TypeError.

Nothing here touches the rollback add-on (``ROLLBACK_SLUG``, the stopped
instance the add-on replaced) except the explicit ``stop`` / ``start``
subcommands and ``seed``, which only reads its state files. The admin token is read from ~/.ha_token (first line)
and never printed. Authentication is attempted ONCE per run: repeated 401s can
get the client IP banned.

Usage:
    uv run python scripts/deploy_addon.py deploy --dry-run
    uv run python scripts/deploy_addon.py deploy [--options opts.json] [--skip-tests]
    uv run python scripts/deploy_addon.py status
    uv run python scripts/deploy_addon.py logs [--lines 300] [--slug SLUG]
    uv run python scripts/deploy_addon.py options opts.json [--no-restart] [--watchdog on|off]
    uv run python scripts/deploy_addon.py export-options opts.json
    uv run python scripts/deploy_addon.py start|stop|restart [--slug SLUG]
    uv run python scripts/deploy_addon.py stage --out DIR
    uv run python scripts/deploy_addon.py seed [--force]
    uv run python scripts/deploy_addon.py restore <backup-dir>
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
ADDON_SRC = REPO_ROOT / "addon" / "battery_optimizer"
APPS_DIR = REPO_ROOT / "appdaemon" / "apps"
ORCHESTRATOR = APPS_DIR / "battery_optimizer.py"
LIB_DIR = APPS_DIR / "battery_optimizer_lib"

HA_HOST = os.environ.get("BO_HA_HOST", "192.168.77.167")
HA_URL = os.environ.get("BO_HA_URL", f"http://{HA_HOST}:8123")
SHARE_ROOT = Path(os.environ.get("BO_SHARE_ROOT", f"//{HA_HOST}"))
ADDON_DIR_NAME = "battery_optimizer"
SLUG = "local_battery_optimizer"
ROLLBACK_SLUG = "a0d7b954_appdaemon"
TOKEN_FILE = Path(os.environ.get("BO_HA_TOKEN_FILE", Path.home() / ".ha_token"))
KEEP_BACKUPS = 5

# Files the stage must contain, relative to its root.
REQUIRED_STAGE_FILES = ("config.yaml", "Dockerfile", "run.sh", "requirements.txt",
                        "DOCS.md", "app/battery_optimizer.py",
                        "app/battery_optimizer_lib/__init__.py",
                        "app/battery_optimizer_lib/ha_host.py",
                        "app/battery_optimizer_lib/addon_main.py")
LF_FILES = ("run.sh", "Dockerfile")
LOG_ERROR_MARKERS = ("Traceback", "ModuleNotFoundError", "TypeError", "ImportError")


class DeployError(Exception):
    pass


def addons_dir(share_root: Path = None) -> Path:
    return Path(share_root or SHARE_ROOT) / "addons" / ADDON_DIR_NAME


def backup_root(share_root: Path = None) -> Path:
    """Outside addons/ by construction; see the module docstring."""
    return Path(share_root or SHARE_ROOT) / "share" / "battery_optimizer_backups"


def app_version() -> str:
    text = ORCHESTRATOR.read_text(encoding="utf-8")
    m = re.search(r'^APP_VERSION = "([^"]+)"', text, re.M)
    if not m:
        raise DeployError("APP_VERSION not found in battery_optimizer.py")
    return m.group(1)


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def git_describe() -> str:
    def run(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True).stdout.strip()

    return f"{run('rev-parse', '--abbrev-ref', 'HEAD')}@{run('rev-parse', '--short', 'HEAD')}"


def git_is_clean() -> bool:
    out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                         capture_output=True, text=True).stdout
    return not out.strip()


def run_tests() -> None:
    log("pytest ...")
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q",
                           "-p", "no:cacheprovider"], cwd=REPO_ROOT,
                          capture_output=True, text=True)
    tail = "\n".join(proc.stdout.strip().splitlines()[-3:])
    log(f"  {tail}")
    if proc.returncode != 0:
        raise DeployError("pytest failed - not deploying")


def compile_all() -> int:
    """Syntax-check every module that ships, without writing .pyc files."""
    files = [ORCHESTRATOR, *sorted(LIB_DIR.glob("*.py"))]
    for path in files:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as e:
            raise DeployError(f"py_compile: {e}")
    return len(files)


# ---------------------------------------------------------------------------
# stage / copy / verify
# ---------------------------------------------------------------------------


def stage(out: Path, version: Optional[str] = None) -> Path:
    """Assemble the add-on build context in ``out`` (created, must be empty)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise DeployError(f"stage directory is not empty: {out}")
    for item in ADDON_SRC.iterdir():
        if item.name in (".gitattributes", "app") or item.name.startswith("__"):
            continue
        if item.is_file():
            shutil.copy2(item, out / item.name)
    app = out / "app"
    (app / "battery_optimizer_lib").mkdir(parents=True)
    shutil.copy2(ORCHESTRATOR, app / ORCHESTRATOR.name)
    for py in sorted(LIB_DIR.glob("*.py")):
        shutil.copy2(py, app / "battery_optimizer_lib" / py.name)
    for name in LF_FILES:
        path = out / name
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    version = version or app_version()
    config = out / "config.yaml"
    text = config.read_text(encoding="utf-8")
    text, n = re.subn(r'^version: .*$', f'version: "{version}"', text, count=1, flags=re.M)
    if n != 1:
        raise DeployError("config.yaml has no version line")
    config.write_text(text, encoding="utf-8")
    missing = [f for f in REQUIRED_STAGE_FILES if not (out / f).is_file()]
    if missing:
        raise DeployError(f"stage is missing {missing}")
    return out


def file_hashes(root: Path) -> Dict[str, str]:
    out = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            out[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def backup(dest: Path, root: Path, keep: int = KEEP_BACKUPS) -> Optional[Path]:
    if not dest.exists():
        return None
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    target = root / f"addon-{stamp}"
    shutil.copytree(dest, target, ignore=shutil.ignore_patterns("__pycache__"))
    backups = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("addon-"))
    for old in backups[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    return target


def sync(stage_dir: Path, dest: Path) -> List[str]:
    """Mirror ``stage_dir`` into ``dest``; returns the pruned paths."""
    dest.mkdir(parents=True, exist_ok=True)
    wanted = file_hashes(stage_dir)
    pruned = []
    for path in sorted(dest.rglob("*"), reverse=True):
        rel = path.relative_to(dest).as_posix()
        if path.is_dir():
            if path.name == "__pycache__":
                shutil.rmtree(path, ignore_errors=True)
            elif not any(path.iterdir()):
                path.rmdir()
            continue
        if rel not in wanted:
            path.unlink()
            pruned.append(rel)
    for rel in wanted:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stage_dir / rel, target)
    now = time.time()
    for rel in wanted:
        os.utime(dest / rel, (now, now))
    got = file_hashes(dest)
    if got != wanted:
        bad = sorted(set(wanted) ^ set(got) | {k for k in wanted if got.get(k) != wanted[k]})
        raise DeployError(f"SHA256 verification failed for {bad}")
    return pruned


# ---------------------------------------------------------------------------
# Supervisor (through HA's websocket, admin token)
# ---------------------------------------------------------------------------


def read_admin_token(path: Path = TOKEN_FILE) -> str:
    try:
        token = path.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError):
        raise DeployError(f"no admin token in {path} (first line)")
    if not token:
        raise DeployError(f"empty admin token in {path}")
    return token


class Supervisor:
    """``supervisor/api`` calls over one authenticated HA websocket."""

    def __init__(self, url: str = HA_URL, token: Optional[str] = None):
        import websocket

        ws_url = url.replace("http://", "ws://").replace("https://", "wss://") + "/api/websocket"
        self._token = token or read_admin_token()
        self._ws = websocket.create_connection(ws_url, timeout=30)
        self._id = 0
        hello = json.loads(self._ws.recv())
        if hello.get("type") != "auth_required":
            raise DeployError(f"unexpected greeting {hello.get('type')}")
        self._ws.send(json.dumps({"type": "auth", "access_token": self._token}))
        reply = json.loads(self._ws.recv())
        if reply.get("type") != "auth_ok":
            # Once only: repeated failures can get this IP banned.
            raise DeployError("Home Assistant refused the admin token - not retrying")

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass

    def api(self, endpoint: str, method: str = "get", data: Optional[dict] = None,
            timeout: int = 300):
        self._id += 1
        msg = {"id": self._id, "type": "supervisor/api", "endpoint": endpoint,
               "method": method, "timeout": timeout}
        if data is not None:
            msg["data"] = data
        self._ws.settimeout(timeout + 30)
        self._ws.send(json.dumps(msg))
        while True:
            reply = json.loads(self._ws.recv())
            if reply.get("id") == self._id and reply.get("type") == "result":
                break
        if not reply.get("success"):
            error = reply.get("error") or {}
            raise DeployError(f"{method.upper()} {endpoint}: {error.get('message', error)}")
        return reply.get("result")

    def info(self, slug: str) -> Optional[dict]:
        try:
            return self.api(f"/addons/{slug}/info")
        except DeployError:
            return None

    def wait_state(self, slug: str, state: str, timeout: float = 180) -> dict:
        deadline = time.monotonic() + timeout
        info = None
        while time.monotonic() < deadline:
            info = self.info(slug) or {}
            if info.get("state") == state:
                return info
            time.sleep(3)
        raise DeployError(f"{slug} did not reach state {state!r} (last: {info and info.get('state')})")


def fetch_logs(slug: str, lines: int = 300, token: Optional[str] = None) -> str:
    """Add-on log via Core's hassio proxy (Range works here, not in PS 5.1)."""
    import requests

    token = token or read_admin_token()
    response = requests.get(
        f"{HA_URL}/api/hassio/addons/{slug}/logs",
        headers={"Authorization": f"Bearer {token}", "Accept": "text/plain",
                 "Range": f"entries=:-{int(lines)}:"},
        timeout=30,
    )
    if response.status_code == 401:
        raise DeployError("logs: 401 - not retrying")
    response.raise_for_status()
    # text/plain without a charset: requests would guess ISO-8859-1 and turn
    # every em dash into mojibake. The add-on writes UTF-8.
    return response.content.decode("utf-8", "replace")


def check_log(text: str, version: Optional[str]) -> List[str]:
    """Problems found in an add-on log excerpt (empty list = clean)."""
    problems = [f"'{m}' in the log" for m in LOG_ERROR_MARKERS if m in text]
    if version and f"Battery Optimizer version {version}" not in text:
        problems.append(f"no 'Battery Optimizer version {version}' line yet")
    return problems


def install_or_update(sup: Supervisor, slug: str = SLUG) -> str:
    sup.api("/store/reload", "post")
    info = sup.info(slug)
    if info is None or not info.get("version"):
        log(f"installing {slug} ...")
        sup.api(f"/store/addons/{slug}/install", "post", timeout=900)
        return "installed"
    if info.get("update_available") or (
        info.get("version_latest") and info.get("version_latest") != info.get("version")
    ):
        log(f"updating {slug} {info.get('version')} -> {info.get('version_latest')} ...")
        sup.api(f"/store/addons/{slug}/update", "post", timeout=900)
        return "updated"
    log(f"rebuilding {slug} {info.get('version')} ...")
    sup.api(f"/addons/{slug}/rebuild", "post", timeout=900)
    return "rebuilt"


def set_options(sup: Supervisor, options: dict, slug: str = SLUG) -> None:
    sup.api(f"/addons/{slug}/options/validate", "post", data={"options": options})
    sup.api(f"/addons/{slug}/options", "post", data={"options": options})


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_stage(args) -> int:
    out = stage(Path(args.out))
    log(f"staged {len(file_hashes(out))} files in {out} (version {app_version()})")
    return 0


def cmd_deploy(args) -> int:
    version = app_version()
    log(f"deploying {git_describe()} as add-on version {version}")
    if not git_is_clean():
        if not args.allow_dirty:
            raise DeployError("working tree is not clean (--allow-dirty to override)")
        log("WARNING: working tree is not clean")
    if not args.skip_tests:
        run_tests()
    log(f"py_compile: {compile_all()} modules OK")
    options = None
    if args.options:
        options = json.loads(Path(args.options).read_text(encoding="utf-8"))
        log(f"options: {len(options)} keys, shadow_mode={options.get('shadow_mode')}, "
            f"device_id={'<set>' if options.get('device_id') else '<empty>'}")

    with tempfile.TemporaryDirectory() as tmp:
        stage_dir = stage(Path(tmp) / "stage", version)
        files = file_hashes(stage_dir)
        dest = addons_dir()
        log(f"target: {dest} ({len(files)} files)")
        if args.dry_run:
            for rel in files:
                log(f"  {rel}")
            log("dry run: nothing written")
            return 0
        saved = backup(dest, backup_root())
        if saved:
            log(f"backup: {saved}")
        pruned = sync(stage_dir, dest)
        log(f"copied + SHA256-verified {len(files)} files"
            + (f", pruned {pruned}" if pruned else ""))

    sup = Supervisor()
    try:
        action = install_or_update(sup)
        if options is not None:
            set_options(sup, options)
            log("options set")
        if args.no_start:
            log(f"{action}; not started (--no-start)")
            return 0
        info = sup.info(SLUG) or {}
        if info.get("state") == "started":
            sup.api(f"/addons/{SLUG}/restart", "post")
        else:
            sup.api(f"/addons/{SLUG}/start", "post")
        info = sup.wait_state(SLUG, "started")
        log(f"{SLUG} {info.get('version')} started")
    finally:
        sup.close()

    deadline = time.monotonic() + 90
    problems = ["not checked"]
    while time.monotonic() < deadline:
        time.sleep(10)
        text = fetch_logs(SLUG, 400)
        problems = check_log(text, version)
        if not problems or any("in the log" in p for p in problems):
            break
    if problems:
        log("post-deploy log check: " + "; ".join(problems))
        return 1
    log("post-deploy log check: version line present, no Traceback/import errors")
    return 0


def cmd_status(args) -> int:
    sup = Supervisor()
    try:
        for slug in (SLUG, ROLLBACK_SLUG):
            info = sup.info(slug)
            if info is None:
                log(f"{slug}: not installed")
                continue
            log(f"{slug}: state={info.get('state')} version={info.get('version')} "
                f"latest={info.get('version_latest')} boot={info.get('boot')} "
                f"arch={info.get('arch')}")
            if slug == SLUG:
                opts = info.get("options") or {}
                log(f"  shadow_mode={opts.get('shadow_mode')} suffix={opts.get('entity_suffix')} "
                    f"device_id={'<set>' if opts.get('device_id') else '<empty>'} "
                    f"options={len(opts)}")
    finally:
        sup.close()
    return 0


def cmd_logs(args) -> int:
    # Bytes, not text: a Windows console codec cannot encode the log's
    # em dashes and degree signs.
    sys.stdout.buffer.write(fetch_logs(args.slug, args.lines).encode("utf-8", "replace"))
    sys.stdout.flush()
    return 0


def cmd_options(args) -> int:
    options = json.loads(Path(args.file).read_text(encoding="utf-8"))
    sup = Supervisor()
    try:
        set_options(sup, options)
        log(f"options set ({len(options)} keys, shadow_mode={options.get('shadow_mode')})")
        if args.watchdog is not None:
            # Supervisor restarts a crashed container only with the watchdog
            # on - the AppDaemon add-on this replaces ran with it.
            sup.api(f"/addons/{SLUG}/options", "post",
                    data={"watchdog": args.watchdog == "on"})
            log(f"watchdog {args.watchdog}")
        if not args.no_restart:
            sup.api(f"/addons/{SLUG}/restart", "post")
            sup.wait_state(SLUG, "started")
            log("restarted")
    finally:
        sup.close()
    return 0


def cmd_export_options(args) -> int:
    """Write the add-on's current options to a JSON file (unredacted)."""
    sup = Supervisor()
    try:
        info = sup.info(args.slug) or {}
    finally:
        sup.close()
    options = info.get("options")
    if not isinstance(options, dict):
        raise DeployError(f"{args.slug}: no options (installed?)")
    Path(args.file).write_text(json.dumps(options, indent=2, sort_keys=True),
                               encoding="utf-8")
    log(f"wrote {len(options)} options to {args.file}")
    return 0


def cmd_lifecycle(args) -> int:
    sup = Supervisor()
    try:
        sup.api(f"/addons/{args.slug}/{args.command}", "post")
        want = "stopped" if args.command == "stop" else "started"
        info = sup.wait_state(args.slug, want)
        log(f"{args.slug}: {info.get('state')}")
    finally:
        sup.close()
    return 0


# Persisted state, by apps.yaml key, with the config loader's defaults.
STATE_FILE_KEYS = {
    "load_profile_file": "/config/load_profile.json",
    "learning_data_file": "",
    "prediction_tracker_file": "/config/prediction_tracker.json",
    "pv_profile_file": "/config/pv_profile.json",
}


def rollback_config_dir(share_root: Path = None) -> Path:
    return Path(share_root or SHARE_ROOT) / "addon_configs" / ROLLBACK_SLUG


def addon_config_dir(share_root: Path = None) -> Path:
    return Path(share_root or SHARE_ROOT) / "addon_configs" / SLUG


def seed_plan(app_args: dict, share_root: Path = None):
    """(key, source, destination) for every state file the live config uses.

    Both containers see their own config dir as ``/config``, so a path is
    mapped by its part below ``/config`` - the add-on keeps the exact paths
    apps.yaml used. A path outside ``/config`` cannot be mapped and is an
    error rather than a guess.
    """
    plan = []
    for key, default in STATE_FILE_KEYS.items():
        path = app_args.get(key, default)
        if not path:
            continue
        if not str(path).startswith("/config/"):
            raise DeployError(f"{key}={path!r} is not under /config")
        rel = str(path)[len("/config/"):]
        plan.append((key, rollback_config_dir(share_root) / rel,
                     addon_config_dir(share_root) / rel))
    return plan


def read_live_app_args(share_root: Path = None) -> dict:
    import yaml

    path = rollback_config_dir(share_root) / "apps" / "apps.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for value in raw.values():
        if isinstance(value, dict) and value.get("module") == "battery_optimizer":
            return value
    raise DeployError(f"no battery_optimizer app in {path}")


def cmd_seed(args) -> int:
    """Copy the rollback instance's JSON state into the add-on's /config."""
    plan = seed_plan(read_live_app_args())
    if not args.force:
        sup = Supervisor()
        try:
            info = sup.info(SLUG) or {}
        finally:
            sup.close()
        if info.get("state") == "started":
            raise DeployError(f"{SLUG} is running and writes these files; stop it "
                              "first (or --force)")
    addon_config_dir().mkdir(parents=True, exist_ok=True)
    for key, src, dst in plan:
        if not src.is_file():
            log(f"  {key}: {src} missing - skipped")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        same = hashlib.sha256(src.read_bytes()).hexdigest() == \
            hashlib.sha256(dst.read_bytes()).hexdigest()
        log(f"  {key}: {src.name} {dst.stat().st_size} bytes "
            f"{'verified' if same else 'CHANGED DURING COPY'}")
    return 0


def cmd_restore(args) -> int:
    source = Path(args.backup)
    if not (source / "config.yaml").is_file():
        raise DeployError(f"{source} does not look like an add-on backup")
    dest = addons_dir()
    sync(source, dest)
    log(f"restored {source} -> {dest}")
    sup = Supervisor()
    try:
        install_or_update(sup)
        sup.api(f"/addons/{SLUG}/restart", "post")
        sup.wait_state(SLUG, "started")
        log("restarted")
    finally:
        sup.close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Deploy the battery optimizer add-on")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("stage")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_stage)

    p = sub.add_parser("deploy")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--options", default=None, help="JSON options to set")
    p.add_argument("--no-start", action="store_true")
    p.set_defaults(func=cmd_deploy)

    p = sub.add_parser("status")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("logs")
    p.add_argument("--slug", default=SLUG)
    p.add_argument("--lines", type=int, default=300)
    p.set_defaults(func=cmd_logs)

    p = sub.add_parser("options")
    p.add_argument("file")
    p.add_argument("--no-restart", action="store_true")
    p.add_argument("--watchdog", choices=("on", "off"), default=None)
    p.set_defaults(func=cmd_options)

    p = sub.add_parser("export-options", help="write the current options to a JSON file")
    p.add_argument("file")
    p.add_argument("--slug", default=SLUG)
    p.set_defaults(func=cmd_export_options)

    for name in ("start", "stop", "restart"):
        p = sub.add_parser(name)
        p.add_argument("--slug", default=SLUG)
        p.set_defaults(func=cmd_lifecycle)

    p = sub.add_parser("seed", help="copy the rollback instance's JSON state into the add-on")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_seed)

    p = sub.add_parser("restore")
    p.add_argument("backup")
    p.set_defaults(func=cmd_restore)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DeployError as e:
        log(f"FAIL: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
