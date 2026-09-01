<#
.SYNOPSIS
    Deploys battery_optimizer.py + battery_optimizer_lib\ to the Home Assistant
    AppDaemon share, following the manual procedure documented in CLAUDE.md
    ("Deployment to the HA machine").

.DESCRIPTION
    AppDaemon hot-reloads on every .py modification, so a multi-file copy into a
    RUNNING add-on is imported while it is still in progress and can load a new
    module against its old peers. This script therefore always: backs up, stops
    the add-on, copies, cleans __pycache__, verifies SHA256, and starts again.

    apps.yaml on the share is LIVE (it holds the HA long-lived token) and is
    NEVER written by this script. It is only read - copied to a temp file for
    the smoke test and deleted straight after.

    Order of operations:
      1. preflight   - git clean check + commit/branch, share reachable
      2. tests       - uv run pytest tests/ -q      (skip with -SkipTests)
      3. compile     - uv run python -m py_compile on every deployed file
      4. smoke test  - scripts\smoke_config.py against the LIVE apps.yaml copy
      5. backup      - <share>\backup-<yyyyMMdd-HHmmss>\ (keeps 5 newest)
      6. stop        - Supervisor API via the HA proxy, or a manual pause
      7. copy        - orchestrator + every *.py of the lib, prune stale *.py
      8. pycache     - remove every __pycache__ under the share apps dir
      9. verify      - SHA256 repo vs share, fail loudly on any mismatch
     10. start       - add-on back up, then tail the log

.PARAMETER Share
    UNC path of the AppDaemon apps directory on the HA machine.

.PARAMETER DryRun
    Run every read-only check (git, tests, compile, smoke test, planning) and
    print exactly what would be copied/removed. Writes nothing to the share,
    creates no backup, never stops the add-on.

.PARAMETER SkipTests
    Skip `uv run pytest`. The compile and smoke-test steps still run.

.PARAMETER AllowDirty
    Deploy even though the git working tree has uncommitted changes. Without it
    a dirty tree is a hard stop, because the commit hash printed (and stored
    next to the backup) would not describe what was actually deployed.

.PARAMETER HaToken
    Long-lived Home Assistant access token. When given, the add-on is stopped
    and started through HA's Supervisor proxy
    (POST <HaUrl>/api/hassio/addons/<slug>/stop|start, GET .../info to poll).
    Without a token the script pauses and asks you to do it in the HA UI.

.PARAMETER HaUrl
    Base URL of Home Assistant (the Supervisor proxy lives under /api/hassio).

.PARAMETER AddonSlug
    Add-on slug. Default a0d7b954_appdaemon.

.PARAMETER NoPause
    Never prompt. Only valid together with -HaToken.

.PARAMETER Restore
    Path of a backup-<timestamp> directory created by this script. Stops the
    add-on, copies the backup back over the share, cleans __pycache__, starts
    the add-on again. Skips git/tests/smoke-test entirely.

.PARAMETER KeepBackups
    How many backup-* directories to keep (default 5, oldest pruned).

.EXAMPLE
    pwsh> .\scripts\deploy.ps1 -DryRun
    Full rehearsal, nothing written.

.EXAMPLE
    pwsh> .\scripts\deploy.ps1 -HaToken $env:HA_TOKEN -NoPause
    Unattended deploy.

.EXAMPLE
    pwsh> .\scripts\deploy.ps1 -Restore '\\192.168.33.167\addon_configs\a0d7b954_appdaemon\apps\backup-20260902-181500'
    Roll back to a previous deploy.

.NOTES
    Windows PowerShell 5.1 compatible: no &&/||, no ternary, no ?? operators.
#>

[CmdletBinding()]
param(
    [string]$Share = '\\192.168.33.167\addon_configs\a0d7b954_appdaemon\apps',
    [switch]$DryRun,
    [switch]$SkipTests,
    [switch]$AllowDirty,
    [string]$HaToken = '',
    [string]$HaUrl = 'http://192.168.33.167:8123',
    [string]$AddonSlug = 'a0d7b954_appdaemon',
    [switch]$NoPause,
    [string]$Restore = '',
    [int]$KeepBackups = 5
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 1.0

$RepoRoot        = Split-Path -Parent $PSScriptRoot
$RepoOrchestrator = Join-Path $RepoRoot 'appdaemon\apps\battery_optimizer.py'
$RepoLib          = Join-Path $RepoRoot 'appdaemon\apps\battery_optimizer_lib'
$SmokeScript      = Join-Path $PSScriptRoot 'smoke_config.py'
$BackupNamePattern = '^backup-\d{8}-\d{6}$'

# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

$script:StepNumber = 0

function Write-Step {
    param([string]$Text)
    $script:StepNumber = $script:StepNumber + 1
    Write-Host ''
    Write-Host ("=== [{0}] {1}" -f $script:StepNumber, $Text) -ForegroundColor Cyan
}

function Write-Info {
    param([string]$Text)
    Write-Host ("    " + $Text)
}

function Write-Ok {
    param([string]$Text)
    Write-Host ("    OK   " + $Text) -ForegroundColor Green
}

function Write-Warn2 {
    param([string]$Text)
    Write-Host ("    WARN " + $Text) -ForegroundColor Yellow
}

function Write-Plan {
    param([string]$Text)
    Write-Host ("    DRY  would " + $Text) -ForegroundColor DarkYellow
}

function Fail {
    param([string]$Text)
    throw $Text
}

function Use-NativeErrorMode {
    # $ErrorActionPreference = 'Stop' plus a host that captures stderr turns any
    # stderr line of a native exe (uv prints progress there) into a terminating
    # NativeCommandError even on exit code 0. Run native tools through here and
    # judge them by $LASTEXITCODE instead.
    param([scriptblock]$Block)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Block
    } finally {
        $ErrorActionPreference = $previous
    }
}

# ---------------------------------------------------------------------------
# Add-on control (HA Supervisor proxy) / manual pause
# ---------------------------------------------------------------------------

function Get-AddonHeaders {
    return @{ 'Authorization' = ('Bearer ' + $HaToken); 'Content-Type' = 'application/json' }
}

function Get-AddonState {
    $uri = ('{0}/api/hassio/addons/{1}/info' -f $HaUrl.TrimEnd('/'), $AddonSlug)
    $response = Invoke-RestMethod -Uri $uri -Method Get -Headers (Get-AddonHeaders) -TimeoutSec 30
    if ($null -eq $response) { return '' }
    if ($null -eq $response.data) { return '' }
    return [string]$response.data.state
}

function Invoke-AddonAction {
    param([ValidateSet('start', 'stop')][string]$Action)
    $uri = ('{0}/api/hassio/addons/{1}/{2}' -f $HaUrl.TrimEnd('/'), $AddonSlug, $Action)
    Write-Info ("POST " + $uri)
    Invoke-RestMethod -Uri $uri -Method Post -Headers (Get-AddonHeaders) -TimeoutSec 120 | Out-Null
}

function Wait-AddonState {
    param([string]$Desired, [int]$TimeoutSeconds = 60)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $state = ''
    while ((Get-Date) -lt $deadline) {
        try {
            $state = Get-AddonState
        } catch {
            $state = ('<info query failed: ' + $_.Exception.Message + '>')
        }
        if ($state -eq $Desired) {
            Write-Ok ("add-on state = " + $Desired)
            return
        }
        Write-Info ("state = '" + $state + "', waiting for '" + $Desired + "' ...")
        Start-Sleep -Seconds 3
    }
    Fail ("add-on did not reach state '" + $Desired + "' within " + $TimeoutSeconds + "s (last state: '" + $state + "')")
}

function Stop-Addon {
    if ($DryRun) {
        Write-Plan ("stop add-on '" + $AddonSlug + "'")
        return
    }
    if ($HaToken -ne '') {
        Invoke-AddonAction -Action 'stop'
        Wait-AddonState -Desired 'stopped' -TimeoutSeconds 60
        return
    }
    Write-Warn2 "No -HaToken given."
    Write-Host ''
    Write-Host ("  ACTION REQUIRED: stop the AppDaemon add-on '" + $AddonSlug + "' in the Home Assistant UI") -ForegroundColor Yellow
    Write-Host ("  (Settings > Add-ons > AppDaemon > Stop), then press Enter here.") -ForegroundColor Yellow
    Read-Host "  Press Enter once the add-on is STOPPED" | Out-Null
}

function Start-Addon {
    if ($DryRun) {
        Write-Plan ("start add-on '" + $AddonSlug + "'")
        return
    }
    if ($HaToken -ne '') {
        Invoke-AddonAction -Action 'start'
        Wait-AddonState -Desired 'started' -TimeoutSeconds 60
        return
    }
    Write-Host ''
    Write-Host ("  ACTION REQUIRED: start the AppDaemon add-on '" + $AddonSlug + "' in the Home Assistant UI,") -ForegroundColor Yellow
    Write-Host ("  then press Enter here.") -ForegroundColor Yellow
    Read-Host "  Press Enter once the add-on is STARTED" | Out-Null
}

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

function Get-RepoLibFiles {
    # Every *.py under battery_optimizer_lib, excluding __pycache__ (and .pyc,
    # which the -Filter already excludes).
    return Get-ChildItem -Path $RepoLib -Recurse -File -Filter '*.py' |
        Where-Object { $_.FullName -notmatch '\\__pycache__\\' } |
        Sort-Object FullName
}

function Get-RelativePath {
    param([string]$Root, [string]$FullPath)
    $rootFull = [System.IO.Path]::GetFullPath($Root)
    if (-not $rootFull.EndsWith('\')) { $rootFull = $rootFull + '\' }
    $full = [System.IO.Path]::GetFullPath($FullPath)
    if ($full.StartsWith($rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $full.Substring($rootFull.Length)
    }
    return $full
}

function Remove-PyCache {
    param([string]$Root)
    if (-not (Test-Path -LiteralPath $Root)) { return }
    $caches = @(Get-ChildItem -Path $Root -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue)
    if ($caches.Count -eq 0) {
        Write-Info "no __pycache__ directories on the share"
        return
    }
    foreach ($cache in $caches) {
        if ($DryRun) {
            Write-Plan ("remove " + $cache.FullName)
        } else {
            Write-Info ("removing " + $cache.FullName)
            Remove-Item -LiteralPath $cache.FullName -Recurse -Force
        }
    }
}

function Copy-FileEnsuringDir {
    param([string]$Source, [string]$Destination)
    $dir = Split-Path -Parent $Destination
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

function Test-SameHash {
    param([string]$A, [string]$B)
    if (-not (Test-Path -LiteralPath $A)) { return $false }
    if (-not (Test-Path -LiteralPath $B)) { return $false }
    $ha = (Get-FileHash -LiteralPath $A -Algorithm SHA256).Hash
    $hb = (Get-FileHash -LiteralPath $B -Algorithm SHA256).Hash
    return ($ha -eq $hb)
}

# ---------------------------------------------------------------------------
# Restore mode
# ---------------------------------------------------------------------------

function Invoke-Restore {
    param([string]$BackupDir)

    Write-Host ''
    Write-Host '########## RESTORE ##########' -ForegroundColor Magenta

    Write-Step "Validate backup directory"
    if (-not (Test-Path -LiteralPath $BackupDir -PathType Container)) {
        Fail ("backup directory not found: " + $BackupDir)
    }
    $backupOrchestrator = Join-Path $BackupDir 'battery_optimizer.py'
    $backupLib = Join-Path $BackupDir 'battery_optimizer_lib'
    if (-not (Test-Path -LiteralPath $backupOrchestrator)) {
        Fail ("backup has no battery_optimizer.py: " + $backupOrchestrator)
    }
    if (-not (Test-Path -LiteralPath $backupLib -PathType Container)) {
        Fail ("backup has no battery_optimizer_lib\: " + $backupLib)
    }
    Write-Ok ("backup looks complete: " + $BackupDir)
    $commitFile = Join-Path $BackupDir 'deployed-commit.txt'
    if (Test-Path -LiteralPath $commitFile) {
        Write-Info "backup was taken when the share held:"
        Get-Content -LiteralPath $commitFile | ForEach-Object { Write-Info ("  " + $_) }
    }

    Write-Step "Share reachable"
    if (-not (Test-Path -LiteralPath $Share -PathType Container)) {
        Fail ("share not reachable: " + $Share)
    }
    Write-Ok $Share

    Write-Step "Stop the add-on"
    Stop-Addon

    Write-Step "Copy the backup back"
    $shareOrchestrator = Join-Path $Share 'battery_optimizer.py'
    $shareLib = Join-Path $Share 'battery_optimizer_lib'
    $backupFiles = @(Get-ChildItem -Path $backupLib -Recurse -File -Filter '*.py' |
        Where-Object { $_.FullName -notmatch '\\__pycache__\\' })
    $backupRelSet = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($file in $backupFiles) {
        [void]$backupRelSet.Add((Get-RelativePath -Root $backupLib -FullPath $file.FullName))
    }
    # Modules added since the backup would otherwise survive the rollback and be
    # imported by the older code.
    $extraFiles = @()
    if (Test-Path -LiteralPath $shareLib -PathType Container) {
        $shareLibFiles = @(Get-ChildItem -Path $shareLib -Recurse -File -Filter '*.py' |
            Where-Object { $_.FullName -notmatch '\\__pycache__\\' })
        foreach ($file in $shareLibFiles) {
            $rel = Get-RelativePath -Root $shareLib -FullPath $file.FullName
            if (-not $backupRelSet.Contains($rel)) { $extraFiles += $file }
        }
    }

    if ($DryRun) {
        Write-Plan ("copy " + $backupOrchestrator + " -> " + $shareOrchestrator)
        foreach ($file in $backupFiles) {
            Write-Plan ("restore battery_optimizer_lib\" + (Get-RelativePath -Root $backupLib -FullPath $file.FullName))
        }
        foreach ($file in $extraFiles) {
            Write-Plan ("delete newer-than-backup " + $file.FullName)
        }
    } else {
        Copy-Item -LiteralPath $backupOrchestrator -Destination $shareOrchestrator -Force
        Write-Info ("restored battery_optimizer.py")
        foreach ($file in $backupFiles) {
            $rel = Get-RelativePath -Root $backupLib -FullPath $file.FullName
            Copy-FileEnsuringDir -Source $file.FullName -Destination (Join-Path $shareLib $rel)
            Write-Info ("restored battery_optimizer_lib\" + $rel)
        }
        foreach ($file in $extraFiles) {
            Write-Info ("deleting (not in the backup) " + $file.FullName)
            Remove-Item -LiteralPath $file.FullName -Force
        }
    }

    Write-Step "Remove __pycache__ on the share"
    Remove-PyCache -Root $Share

    Write-Step "Start the add-on"
    Start-Addon

    Write-Host ''
    Write-Host 'RESTORE COMPLETE.' -ForegroundColor Green
    Write-Host 'Watch the AppDaemon log for ModuleNotFoundError / TypeError in the first minute.' -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------

function Invoke-Deploy {

    Write-Host ''
    if ($DryRun) {
        Write-Host '########## DEPLOY (DRY RUN - nothing will be written) ##########' -ForegroundColor Magenta
    } else {
        Write-Host '########## DEPLOY ##########' -ForegroundColor Magenta
    }
    Write-Info ("repo  : " + $RepoRoot)
    Write-Info ("share : " + $Share)

    # -- 1. preflight: git ---------------------------------------------------
    Write-Step "Preflight: git state"
    $status = Use-NativeErrorMode { & git -C $RepoRoot status --porcelain }
    if ($LASTEXITCODE -ne 0) { Fail "git status failed - is this a git repo?" }
    $dirty = @($status | Where-Object { $_ -ne '' })
    $commit = (Use-NativeErrorMode { & git -C $RepoRoot rev-parse HEAD }).Trim()
    $shortCommit = (Use-NativeErrorMode { & git -C $RepoRoot rev-parse --short HEAD }).Trim()
    $branch = (Use-NativeErrorMode { & git -C $RepoRoot rev-parse --abbrev-ref HEAD }).Trim()
    Write-Info ("branch : " + $branch)
    Write-Info ("commit : " + $shortCommit + "  (" + $commit + ")")
    if ($dirty.Count -gt 0) {
        Write-Warn2 ($dirty.Count.ToString() + " uncommitted change(s):")
        foreach ($line in $dirty) { Write-Info ("  " + $line) }
        if (-not $AllowDirty) {
            Fail ("working tree is dirty - commit/stash first, or pass -AllowDirty " +
                  "(the recorded commit would not describe what is deployed)")
        }
        Write-Warn2 "-AllowDirty given: deploying a tree that does NOT match the commit above."
    } else {
        Write-Ok "working tree clean"
    }

    # -- 2. preflight: share -------------------------------------------------
    Write-Step "Preflight: share reachable"
    if (-not (Test-Path -LiteralPath $Share -PathType Container)) {
        Fail ("share not reachable: " + $Share)
    }
    Write-Ok $Share
    $shareAppsYaml = Join-Path $Share 'apps.yaml'
    if (-not (Test-Path -LiteralPath $shareAppsYaml)) {
        Fail ("LIVE apps.yaml not found on the share: " + $shareAppsYaml)
    }
    Write-Ok ("live apps.yaml present (read-only to this script): " + $shareAppsYaml)
    if (-not (Test-Path -LiteralPath $RepoOrchestrator)) { Fail ("missing " + $RepoOrchestrator) }
    if (-not (Test-Path -LiteralPath $RepoLib -PathType Container)) { Fail ("missing " + $RepoLib) }

    # -- 3. tests ------------------------------------------------------------
    Write-Step "Unit tests"
    if ($SkipTests) {
        Write-Warn2 "-SkipTests given: pytest NOT run."
    } else {
        Push-Location $RepoRoot
        try {
            Use-NativeErrorMode { & uv run pytest tests/ -q }
            if ($LASTEXITCODE -ne 0) { Fail ("pytest failed (exit " + $LASTEXITCODE + ")") }
        } finally {
            Pop-Location
        }
        Write-Ok "pytest passed"
    }

    # -- 4. compile ----------------------------------------------------------
    Write-Step "Byte-compile the files that will be deployed"
    $repoLibFiles = @(Get-RepoLibFiles)
    if ($repoLibFiles.Count -eq 0) { Fail "no *.py found in battery_optimizer_lib" }
    $compileArgs = @('run', 'python', '-m', 'py_compile', $RepoOrchestrator)
    foreach ($file in $repoLibFiles) { $compileArgs += $file.FullName }
    Push-Location $RepoRoot
    try {
        Use-NativeErrorMode { & uv @compileArgs }
        if ($LASTEXITCODE -ne 0) { Fail ("py_compile failed (exit " + $LASTEXITCODE + ")") }
    } finally {
        Pop-Location
    }
    Write-Ok ("compiled battery_optimizer.py + " + $repoLibFiles.Count + " library modules")

    # -- 5. smoke test against the LIVE apps.yaml ----------------------------
    Write-Step "Smoke test against the LIVE apps.yaml"
    if (-not (Test-Path -LiteralPath $SmokeScript)) { Fail ("missing " + $SmokeScript) }
    $tempYaml = Join-Path $env:TEMP ('battery-optimizer-smoke-' + [guid]::NewGuid().ToString('N') + '.yaml')
    Write-Info ("copying the live apps.yaml to " + $tempYaml + " (deleted right after)")
    Copy-Item -LiteralPath $shareAppsYaml -Destination $tempYaml -Force
    Push-Location $RepoRoot
    try {
        Use-NativeErrorMode { & uv run python $SmokeScript $tempYaml }
        $smokeExit = $LASTEXITCODE
    } finally {
        Pop-Location
        Remove-Item -LiteralPath $tempYaml -Force -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $tempYaml) {
            Write-Warn2 ("could not delete the temp apps.yaml copy: " + $tempYaml + " - DELETE IT MANUALLY, it holds the HA token")
        } else {
            Write-Info "temp apps.yaml copy deleted"
        }
    }
    if ($smokeExit -ne 0) {
        Fail ("smoke test failed (exit " + $smokeExit + ") - the LIVE config does not load with this code")
    }
    Write-Ok "the live apps.yaml loads with this code"

    # -- 6. plan the copy ----------------------------------------------------
    Write-Step "Plan"
    $shareOrchestrator = Join-Path $Share 'battery_optimizer.py'
    $shareLib = Join-Path $Share 'battery_optimizer_lib'

    $plannedCopies = @()
    $plannedCopies += [pscustomobject]@{ Source = $RepoOrchestrator; Destination = $shareOrchestrator; Rel = 'battery_optimizer.py' }
    foreach ($file in $repoLibFiles) {
        $rel = Get-RelativePath -Root $RepoLib -FullPath $file.FullName
        $plannedCopies += [pscustomobject]@{
            Source      = $file.FullName
            Destination = (Join-Path $shareLib $rel)
            Rel         = ('battery_optimizer_lib\' + $rel)
        }
    }

    $repoRelSet = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($file in $repoLibFiles) {
        [void]$repoRelSet.Add((Get-RelativePath -Root $RepoLib -FullPath $file.FullName))
    }
    $staleFiles = @()
    if (Test-Path -LiteralPath $shareLib -PathType Container) {
        $shareLibFiles = @(Get-ChildItem -Path $shareLib -Recurse -File -Filter '*.py' |
            Where-Object { $_.FullName -notmatch '\\__pycache__\\' })
        foreach ($file in $shareLibFiles) {
            $rel = Get-RelativePath -Root $shareLib -FullPath $file.FullName
            if (-not $repoRelSet.Contains($rel)) { $staleFiles += $file }
        }
    }

    Write-Info ("files to copy (" + $plannedCopies.Count + "):")
    foreach ($item in $plannedCopies) {
        $mark = '  new '
        if (Test-Path -LiteralPath $item.Destination) {
            if (Test-SameHash -A $item.Source -B $item.Destination) {
                $mark = '  same'
            } else {
                $mark = '  diff'
            }
        }
        Write-Info ($mark + '  ' + $item.Rel)
    }
    if ($staleFiles.Count -gt 0) {
        Write-Warn2 ("stale *.py on the share, not present in the repo (" + $staleFiles.Count + "):")
        foreach ($file in $staleFiles) { Write-Info ("  del   " + $file.FullName) }
    } else {
        Write-Ok "no stale *.py on the share"
    }

    # -- 7. backup -----------------------------------------------------------
    Write-Step "Backup the current share state"
    $timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $backupDir = Join-Path $Share ('backup-' + $timestamp)
    if ($DryRun) {
        Write-Plan ("create " + $backupDir + " with battery_optimizer.py + battery_optimizer_lib\ + deployed-commit.txt")
    } else {
        New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
        if (Test-Path -LiteralPath $shareOrchestrator) {
            Copy-Item -LiteralPath $shareOrchestrator -Destination (Join-Path $backupDir 'battery_optimizer.py') -Force
        } else {
            Write-Warn2 "share has no battery_optimizer.py to back up (first deploy?)"
        }
        if (Test-Path -LiteralPath $shareLib -PathType Container) {
            Copy-Item -LiteralPath $shareLib -Destination $backupDir -Recurse -Force
            $backedUpCache = @(Get-ChildItem -Path (Join-Path $backupDir 'battery_optimizer_lib') -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue)
            foreach ($cache in $backedUpCache) { Remove-Item -LiteralPath $cache.FullName -Recurse -Force }
        } else {
            Write-Warn2 "share has no battery_optimizer_lib\ to back up (first deploy?)"
        }
        $commitLines = @(
            ('deployed-commit : ' + $commit),
            ('short           : ' + $shortCommit),
            ('branch          : ' + $branch),
            ('deployed-at     : ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss K')),
            ('deployed-by     : ' + $env:USERNAME + '@' + $env:COMPUTERNAME),
            ('dirty-tree      : ' + ($dirty.Count -gt 0))
        )
        Set-Content -LiteralPath (Join-Path $backupDir 'deployed-commit.txt') -Value $commitLines -Encoding utf8
        Write-Ok ("backup created: " + $backupDir)
    }

    # prune old backups
    $existingBackups = @(Get-ChildItem -Path $Share -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match $BackupNamePattern } |
        Sort-Object Name -Descending)
    if ($DryRun) {
        # the (not created) new backup would occupy one of the kept slots
        $keep = $KeepBackups - 1
    } else {
        $keep = $KeepBackups
    }
    if ($keep -lt 0) { $keep = 0 }
    $toPrune = @($existingBackups | Select-Object -Skip $keep)
    if ($toPrune.Count -eq 0) {
        Write-Info ("backup retention: " + $existingBackups.Count + " backup dir(s) present, keeping " + $KeepBackups + " - nothing to prune")
    } else {
        foreach ($old in $toPrune) {
            if ($DryRun) {
                Write-Plan ("prune old backup " + $old.FullName)
            } else {
                Write-Info ("pruning old backup " + $old.FullName)
                Remove-Item -LiteralPath $old.FullName -Recurse -Force
            }
        }
    }

    # -------- everything below touches the running add-on --------------------
    $stopped = $false
    try {
        # -- 8. stop ---------------------------------------------------------
        Write-Step "Stop the AppDaemon add-on"
        Write-Info "AppDaemon hot-reloads on every .py write; a multi-file copy into a running add-on can import a new module against its old peers."
        Stop-Addon
        $stopped = $true

        # -- 9. copy ---------------------------------------------------------
        Write-Step "Copy"
        foreach ($item in $plannedCopies) {
            if ($DryRun) {
                Write-Plan ("copy " + $item.Rel)
            } else {
                Copy-FileEnsuringDir -Source $item.Source -Destination $item.Destination
                Write-Info ("copied " + $item.Rel)
            }
        }
        foreach ($file in $staleFiles) {
            if ($DryRun) {
                Write-Plan ("delete stale " + $file.FullName)
            } else {
                Write-Info ("deleting stale " + $file.FullName)
                Remove-Item -LiteralPath $file.FullName -Force
            }
        }

        # -- 10. pycache -----------------------------------------------------
        Write-Step "Remove __pycache__ on the share"
        Remove-PyCache -Root $Share

        # -- 11. verify ------------------------------------------------------
        Write-Step "Verify (SHA256 repo vs share)"
        if ($DryRun) {
            Write-Plan ("verify SHA256 of " + $plannedCopies.Count + " file(s)")
        } else {
            $mismatches = @()
            foreach ($item in $plannedCopies) {
                if (-not (Test-SameHash -A $item.Source -B $item.Destination)) {
                    $mismatches += $item.Rel
                }
            }
            if ($mismatches.Count -gt 0) {
                Write-Host ''
                Write-Host ("    MISMATCH on " + $mismatches.Count + " file(s):") -ForegroundColor Red
                foreach ($rel in $mismatches) { Write-Host ("      " + $rel) -ForegroundColor Red }
                Fail ("SHA256 verification failed for " + $mismatches.Count + " file(s)")
            }
            Write-Ok ($plannedCopies.Count.ToString() + " file(s) verified identical to the repo")
            Write-Ok ("deployed commit " + $shortCommit + " (" + $branch + "); recorded in " + (Join-Path $backupDir 'deployed-commit.txt'))
        }

        # -- 12. start -------------------------------------------------------
        Write-Step "Start the AppDaemon add-on"
        Start-Addon
    } catch {
        Write-Host ''
        Write-Host '################ DEPLOY FAILED ################' -ForegroundColor Red
        Write-Host ("  " + $_.Exception.Message) -ForegroundColor Red
        if ($stopped -and -not $DryRun) {
            Write-Host ''
            Write-Host ("  The add-on was STOPPED and the share may be half-updated.") -ForegroundColor Red
            Write-Host ("  Backup of the previous state:") -ForegroundColor Yellow
            Write-Host ("    " + $backupDir) -ForegroundColor Yellow
            Write-Host ("  Restore it with:") -ForegroundColor Yellow
            $restoreCmd = ('    .\scripts\deploy.ps1 -Restore "' + $backupDir + '"')
            if ($HaToken -ne '') { $restoreCmd = $restoreCmd + ' -HaToken $env:HA_TOKEN -NoPause' }
            Write-Host $restoreCmd -ForegroundColor Yellow
        }
        throw
    }

    Write-Host ''
    if ($DryRun) {
        Write-Host 'DRY RUN COMPLETE - nothing was written to the share.' -ForegroundColor Green
    } else {
        Write-Host ('DEPLOY COMPLETE - commit ' + $shortCommit + ' on ' + $branch) -ForegroundColor Green
        Write-Host ''
        Write-Host 'Now tail the AppDaemon log for the first minute and look for:' -ForegroundColor Yellow
        Write-Host '  ModuleNotFoundError   (a module was imported against stale peers)' -ForegroundColor Yellow
        Write-Host '  TypeError             (a signature changed but a caller did not)' -ForegroundColor Yellow
        Write-Host 'HA UI: Settings > Add-ons > AppDaemon > Log.' -ForegroundColor Yellow
        Write-Host ('Roll back with: .\scripts\deploy.ps1 -Restore "' + $backupDir + '"') -ForegroundColor Yellow
    }
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if ($NoPause -and $HaToken -eq '') {
    Write-Host "-NoPause requires -HaToken (without a token the add-on must be stopped/started by hand)." -ForegroundColor Red
    exit 2
}

try {
    if ($Restore -ne '') {
        Invoke-Restore -BackupDir $Restore
    } else {
        Invoke-Deploy
    }
    exit 0
} catch {
    Write-Host ''
    Write-Host ('ERROR: ' + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
