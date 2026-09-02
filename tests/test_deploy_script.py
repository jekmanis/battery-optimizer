"""
Static guards on scripts/deploy.ps1.

These are not "does PowerShell work" tests - they encode the one deployment
invariant that a green SHA256 verification cannot protect:

    NOTHING but battery_optimizer.py, battery_optimizer_lib/ and explicitly
    allowed extra apps may hold *.py under the AppDaemon apps directory.

AppDaemon 4.5 discovers apps with ``app_dir.rglob("*.py")`` and calls
``sys.path.insert(0, <dir>)`` for every directory that contains .py files and
has no ``__init__.py``.  Such a directory therefore lands at the FRONT of
sys.path.  On 2026-09-02 deploy.ps1 wrote its backup to
``apps/backup-20260902-015911/`` and AppDaemon imported *the backup* -- it ran
the previous commit while the SHA256 check on ``apps/`` passed, and the only
visible symptom was an old log wording and a health sensor missing the
attributes the new commit added.

The tests below assert that the script cannot regress into that shape.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_PS1 = REPO_ROOT / "scripts" / "deploy.ps1"
SCRIPTS_README = REPO_ROOT / "scripts" / "README.md"


@pytest.fixture(scope="module")
def deploy_text():
    assert DEPLOY_PS1.is_file(), "scripts/deploy.ps1 is missing"
    return DEPLOY_PS1.read_text(encoding="utf-8")


def code_lines(text):
    """Script lines with comment-only and doc-comment lines removed."""
    out = []
    in_block_comment = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("<#"):
            in_block_comment = True
        if in_block_comment:
            if "#>" in stripped:
                in_block_comment = False
            continue
        if stripped.startswith("#"):
            continue
        out.append(line)
    return out


# --------------------------------------------------------------------------
# The backup must not live inside the apps directory
# --------------------------------------------------------------------------

def test_backup_dir_is_not_built_from_the_apps_share(deploy_text):
    """The 2026-09-02 bug verbatim: Join-Path $Share ('backup-' + $timestamp)."""
    offenders = [
        line for line in code_lines(deploy_text)
        if re.search(r"Join-Path\s+\$Share\s+\(\s*'backup-", line)
    ]
    assert not offenders, (
        "the timestamped backup is being created inside the apps directory; "
        "AppDaemon will import from it and shadow the deployed package: %r" % offenders
    )


def test_backup_root_defaults_outside_the_apps_directory(deploy_text):
    assert "$BackupRoot" in deploy_text, "deploy.ps1 has no $BackupRoot"
    # Default is derived from the PARENT of the apps directory.
    assert re.search(
        r"\$ShareRoot\s*=\s*Split-Path\s+-Parent\s+\$Share", deploy_text
    ), "the backup root must be derived from the parent of -Share, not from -Share"
    assert "'backups'" in deploy_text and "'battery_optimizer'" in deploy_text, (
        "expected the default backup root <share-root>\\backups\\battery_optimizer"
    )


def test_backup_root_inside_apps_is_refused(deploy_text):
    assert "function Test-PathInside" in deploy_text
    assert re.search(
        r"if\s*\(Test-PathInside\s+-Root\s+\$Share\s+-Candidate\s+\$BackupRoot\)",
        deploy_text,
    ), "-BackupRoot pointing inside the apps directory must be rejected"


def test_backup_pruning_scans_the_backup_root_not_the_share(deploy_text):
    prune = re.search(
        r"#\s*prune old backups\s*\n\s*\$existingBackups\s*=\s*@\(Get-ChildItem\s+-Path\s+(\S+)",
        deploy_text,
    )
    assert prune, "could not find the backup pruning block"
    assert prune.group(1) == "$BackupRoot", (
        "pruning must enumerate the backup root, not the live apps directory "
        "(found %s)" % prune.group(1)
    )


# --------------------------------------------------------------------------
# Pre-flight scan for anything that shadows the deployed package
# --------------------------------------------------------------------------

def test_stray_scan_exists_and_runs_before_the_copy(deploy_text):
    assert "function Get-StrayApps" in deploy_text
    assert "function Invoke-StrayScan" in deploy_text
    scan_at = deploy_text.index("Invoke-StrayScan -AppsDir $Share")
    copy_at = deploy_text.index("Write-Step \"Copy\"")
    backup_at = deploy_text.index("Write-Step \"Backup the current share state")
    assert scan_at < backup_at < copy_at, (
        "the stray scan must run before the backup and the copy"
    )


def test_stray_scan_allows_only_our_files_and_the_allow_list(deploy_text):
    scan = deploy_text[deploy_text.index("function Get-StrayApps"):]
    scan = scan[:scan.index("function Move-StrayBackupDir")]
    assert "'battery_optimizer.py'" in scan
    assert "'battery_optimizer_lib'" in scan
    assert "'__pycache__'" in scan
    assert "$Allowed" in scan, "the extra-apps allow list must be honoured"
    assert "-Recurse" in scan, "subdirectories must be scanned recursively"


def test_allowed_apps_default_is_hello_py(deploy_text):
    assert re.search(
        r"\[string\[\]\]\$AllowedApps\s*=\s*@\('hello\.py'\)", deploy_text
    ), "the default allow list should be hello.py (the stock AppDaemon sample app)"


def test_stray_scan_aborts_without_move_stray_backups(deploy_text):
    scan = deploy_text[deploy_text.index("function Invoke-StrayScan"):]
    assert "$MoveStrayBackups" in scan
    assert "Fail (" in scan, "a stray path must abort the deploy"
    assert "backup-20260902-015911" in scan, (
        "the abort message should name the production case so the operator "
        "understands why a green SHA256 run still ran the wrong code"
    )


def test_stray_scan_also_runs_in_dry_run(deploy_text):
    """The scan is unconditional; only the mutating move is $DryRun-gated."""
    deploy_fn = deploy_text[deploy_text.index("function Invoke-Deploy"):]
    call = re.search(r"^\s*Invoke-StrayScan -AppsDir \$Share\s*$", deploy_fn, re.M)
    assert call, "Invoke-StrayScan must be called unconditionally in Invoke-Deploy"


def test_stray_backups_are_moved_never_deleted(deploy_text):
    move = deploy_text[deploy_text.index("function Move-StrayBackupDir"):]
    move = move[:move.index("function Invoke-StrayScan")]
    assert "Move-Item" in move
    assert "Remove-Item" not in move, (
        "a stray backup is the only copy of the previous deploy - it is moved, never deleted"
    )


# --------------------------------------------------------------------------
# Restore
# --------------------------------------------------------------------------

def test_restore_evacuates_backups_left_inside_apps(deploy_text):
    restore = deploy_text[deploy_text.index("function Invoke-Restore"):]
    restore = restore[:restore.index("function Invoke-Deploy")]
    assert "Get-StrayApps" in restore, (
        "after restoring, the apps directory must be re-checked for backup dirs"
    )
    assert "Move-StrayBackupDir" in restore, (
        "a backup left inside apps\\ after a rollback shadows the restored code"
    )
    # The move must happen before the add-on is started again.
    assert restore.index("Move-StrayBackupDir") < restore.index("Start-Addon")


def test_restore_accepts_an_in_apps_backup_path(deploy_text):
    restore = deploy_text[deploy_text.index("function Invoke-Restore"):]
    restore = restore[:restore.index("function Invoke-Deploy")]
    assert "Test-PathInside -Root $Share -Candidate $BackupDir" in restore, (
        "restoring from an old in-apps backup must be accepted (and flagged)"
    )


# --------------------------------------------------------------------------
# Post-deploy verification and mtimes
# --------------------------------------------------------------------------

def test_post_deploy_check_reads_the_addon_log(deploy_text):
    assert "function Invoke-PostDeployCheck" in deploy_text
    assert "/api/hassio/addons/{1}/logs?lines={2}" in deploy_text
    check = deploy_text[deploy_text.index("function Invoke-PostDeployCheck"):]
    assert "Initializing Battery Optimizer" in check
    assert "ModuleNotFoundError" in check
    assert "worker threads?" in check, "report the AppDaemon worker-thread count"


def test_post_deploy_check_is_best_effort(deploy_text):
    check = deploy_text[deploy_text.index("function Invoke-PostDeployCheck"):]
    end = check.index("\n# ---")  # the next top-level section banner
    check = check[:end]
    assert "Fail (" not in check, (
        "the post-deploy check must warn, never fail the deploy (the HA API may "
        "be unreachable from the deploying machine)"
    )


def test_deployed_files_are_stamped_with_the_deploy_time(deploy_text):
    assert "function Set-DeployTimestamp" in deploy_text
    assert "LastWriteTime = $When" in deploy_text, (
        "Copy-Item preserves the source mtime, so the copied files must be touched"
    )
    stamp_at = deploy_text.index("Set-DeployTimestamp -Paths $destinations")
    verify_at = deploy_text.index("SHA256 verification failed for")
    assert verify_at < stamp_at, "stamp the files only after the hash verification"


def test_last_deploy_marker_is_written_outside_apps(deploy_text):
    assert "$LastDeployFile = Join-Path $BackupRoot 'last-deploy.txt'" in deploy_text
    assert "Set-Content -LiteralPath $LastDeployFile" in deploy_text


# --------------------------------------------------------------------------
# Invariants inherited from the manual procedure
# --------------------------------------------------------------------------

def test_live_apps_yaml_is_never_written(deploy_text):
    for line in code_lines(deploy_text):
        if "$shareAppsYaml" not in line:
            continue
        assert not re.search(r"(Set-Content|Out-File|Add-Content)", line), (
            "apps.yaml on the share holds the HA token and is read-only: %r" % line
        )
        # It may only appear as a Copy-Item SOURCE.
        if "Copy-Item" in line:
            assert re.search(r"Copy-Item\s+-LiteralPath\s+\$shareAppsYaml", line), line


def test_addon_is_stopped_before_the_copy(deploy_text):
    deploy_fn = deploy_text[deploy_text.index("function Invoke-Deploy"):]
    stop_at = deploy_fn.index("Write-Step \"Stop the AppDaemon add-on\"")
    copy_at = deploy_fn.index("Write-Step \"Copy\"")
    assert stop_at < copy_at


def test_no_powershell_7_only_syntax(deploy_text):
    """Windows PowerShell 5.1: no &&/||, no ternary, no ?? / ?."""
    for number, line in enumerate(code_lines(deploy_text), start=1):
        assert "&&" not in line, "line %d uses && (not PS 5.1): %r" % (number, line)
        assert "||" not in line, "line %d uses || (not PS 5.1): %r" % (number, line)
        assert "??" not in line, "line %d uses ?? (not PS 5.1): %r" % (number, line)


@pytest.mark.skipif(
    shutil.which("powershell") is None, reason="Windows PowerShell not available"
)
def test_deploy_script_parses():
    result = subprocess.run(
        [
            "powershell", "-NoProfile", "-Command",
            "[scriptblock]::Create((Get-Content -Raw '%s')) | Out-Null; 'PARSE OK'"
            % DEPLOY_PS1,
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PARSE OK" in result.stdout


# --------------------------------------------------------------------------
# Documentation
# --------------------------------------------------------------------------

def test_readme_documents_the_shadowing_hazard():
    assert SCRIPTS_README.is_file()
    text = SCRIPTS_README.read_text(encoding="utf-8")
    assert "backups/battery_optimizer" in text.replace("\\", "/"), (
        "the README must name the new backup location"
    )
    assert "-MoveStrayBackups" in text
    assert "sys.path" in text, "explain WHY a backup under apps/ is dangerous"
    assert "backup-20260902-015911" in text, "name the production case"
