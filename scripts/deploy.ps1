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

    NOTHING that contains *.py may live under the apps directory except
    battery_optimizer.py, battery_optimizer_lib\ and the explicitly allowed
    extra apps (-AllowedApps, default hello.py). AppDaemon 4.5 discovers apps
    with app_dir.rglob("*.py") and does sys.path.insert(0, <dir>) for EVERY
    subdirectory that holds .py files and has no __init__.py, so such a
    directory lands at the FRONT of sys.path and shadows the real package.
    That is exactly what a backup directory inside apps\ did on 2026-09-02:
    apps\backup-20260902-015911\ won the import race and AppDaemon ran the
    PREVIOUS commit while SHA256 verification of apps\ passed. Backups
    therefore live OUTSIDE the apps directory, in
    <share-root>\backups\battery_optimizer\.

    apps.yaml on the share is LIVE (it holds the HA long-lived token) and is
    NEVER written by this script. It is only read - copied to a temp file for
    the smoke test and deleted straight after.

    Order of operations:
      1. preflight   - git clean check + commit/branch, share reachable
      2. strays      - no shadowing *.py / directories under the apps dir
      3. tests       - uv run pytest tests/ -q      (skip with -SkipTests)
      4. compile     - uv run python -m py_compile on every deployed file
      5. smoke test  - scripts\smoke_config.py against the LIVE apps.yaml copy
      6. backup      - <share-root>\backups\battery_optimizer\backup-<ts>\
                       (keeps 5 newest) + last-deploy.txt
      7. stop        - the hassio.addon_stop service (default), HA's Supervisor
                       proxy (-AddonApi proxy, legacy), or a manual pause
      8. copy        - orchestrator + every *.py of the lib, prune stale *.py
      9. pycache     - remove every __pycache__ under the share apps dir
     10. verify      - SHA256 repo vs share, fail loudly on any mismatch
     11. touch       - stamp the copied files with the deploy time
     12. start       - add-on back up
     13. postcheck   - best-effort: read the add-on log back and confirm the
                       app initialized without an import error

.PARAMETER Share
    UNC path of the AppDaemon apps directory on the HA machine.

.PARAMETER BackupRoot
    Where backup-<ts> directories are written. Default:
    <parent of -Share>\backups\battery_optimizer. It MUST NOT be inside the
    apps directory - see the shadowing note above; the script refuses such a
    value.

.PARAMETER AllowedApps
    File names of other AppDaemon apps that legitimately live directly in the
    apps directory. Default: hello.py. Anything else with a .py extension
    aborts the deploy.

.PARAMETER MoveStrayBackups
    When the stray scan finds backup-* directories inside the apps directory,
    MOVE them to -BackupRoot instead of aborting. They are never deleted.

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
    and started through the API selected by -AddonApi, and the post-deploy
    check reads the add-on log and sensor.battery_optimizer back. Without a
    token the script pauses and asks you to do it in the HA UI.

    The default -AddonApi service and both post-deploy reads work with ANY
    authenticated user's token - admin rights are not needed. On this machine
    the token is in $env:HA_TOKEN (a user environment variable, also kept in
    ~\.ha_token).

.PARAMETER HaUrl
    Base URL of Home Assistant (the Supervisor proxy lives under /api/hassio).

.PARAMETER AddonSlug
    Add-on slug. Default a0d7b954_appdaemon.

.PARAMETER AddonApi
    Which API stops/starts the add-on.

      service (default) POST <HaUrl>/api/services/hassio/addon_stop|addon_start
              with the body {"addon": "<slug>"}. Any authenticated user may
              call a service, so this needs no admin rights. The add-on state
              cannot be queried this way, so it is inferred from AppDaemon's
              own HTTP server (-AppDaemonPort), which listens exactly while
              the add-on runs: port refused = stopped, port accepts = started.

      proxy   LEGACY, for older HA versions. POST
              <HaUrl>/api/hassio/addons/<slug>/stop|start and poll
              GET .../info for the state. On current HA (2026.8) the REST
              Supervisor proxy no longer forwards add-on info/start/stop for
              ANY token - admin included: GET /api/hassio/addons/<slug>/info
              answers HTTP 401. The proxy only forwards a small allowlist
              (logs, ingress, backup upload/download); add-on control moved to
              the admin-only websocket command supervisor/api, which this
              script deliberately does not implement.

    The add-on LOG endpoint (GET /api/hassio/addons/<slug>/logs) is on that
    allowlist and answers 200 for an admin and a non-admin token alike, so the
    post-deploy log check runs in BOTH modes.

    -AddonApi service requires -HaToken (no token, no service call); pass no
    -AddonApi at all and no -HaToken to stop/start the add-on by hand instead.

.PARAMETER AppDaemonPort
    TCP port of AppDaemon's own HTTP server on the HA machine, used as the
    add-on up/down signal in the default -AddonApi service mode. Default 5050.

.PARAMETER NoPause
    Never prompt. Only valid together with -HaToken.

.PARAMETER Restore
    Path of a backup-<timestamp> directory created by this script, in either
    the new location (<share-root>\backups\battery_optimizer\) or an old
    in-apps one. Stops the add-on, copies the backup back over the share,
    evacuates any backup directory still sitting inside apps\, cleans
    __pycache__, starts the add-on again. Skips git/tests/smoke-test entirely.

.PARAMETER KeepBackups
    How many backup-* directories to keep (default 5, oldest pruned).

.PARAMETER SkipPostCheck
    Do not read the add-on log back after starting it.

.PARAMETER PostCheckTimeoutSeconds
    How long to poll the add-on log for the "Initializing Battery Optimizer"
    line (default 90).

.EXAMPLE
    pwsh> .\scripts\deploy.ps1 -DryRun
    Full rehearsal, nothing written.

.EXAMPLE
    pwsh> .\scripts\deploy.ps1 -HaToken $env:HA_TOKEN -NoPause
    Unattended deploy. Stop/start through hassio.addon_stop /
    hassio.addon_start, add-on state probed on the AppDaemon port, then the
    add-on log and sensor.battery_optimizer are read back. Any authenticated
    user's token works; on this machine $env:HA_TOKEN is already set.

.EXAMPLE
    pwsh> .\scripts\deploy.ps1 -HaToken $env:HA_TOKEN -NoPause -AddonApi proxy
    Same, but stop/start through HA's Supervisor proxy. Only for HA versions
    old enough to still forward add-on start/stop there.

.EXAMPLE
    pwsh> .\scripts\deploy.ps1 -MoveStrayBackups -HaToken $env:HA_TOKEN -NoPause
    Evacuate legacy backup-* directories out of apps\ first, then deploy.

.EXAMPLE
    pwsh> .\scripts\deploy.ps1 -Restore '\\192.168.33.167\addon_configs\a0d7b954_appdaemon\backups\battery_optimizer\backup-20260902-181500'
    Roll back to a previous deploy.

.NOTES
    Windows PowerShell 5.1 compatible: no &&/||, no ternary, no ?? operators.
#>

[CmdletBinding()]
param(
    [string]$Share = '\\192.168.33.167\addon_configs\a0d7b954_appdaemon\apps',
    [string]$BackupRoot = '',
    [switch]$DryRun,
    [switch]$SkipTests,
    [switch]$AllowDirty,
    [string[]]$AllowedApps = @('hello.py'),
    [switch]$MoveStrayBackups,
    [string]$HaToken = '',
    [string]$HaUrl = 'http://192.168.33.167:8123',
    [string]$AddonSlug = 'a0d7b954_appdaemon',
    [ValidateSet('proxy', 'service')]
    [string]$AddonApi = 'service',
    [int]$AppDaemonPort = 5050,
    [switch]$NoPause,
    [string]$Restore = '',
    [int]$KeepBackups = 5,
    [switch]$SkipPostCheck,
    [int]$PostCheckTimeoutSeconds = 90
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 1.0

# Was -AddonApi typed, or is this just the default? Without a token the script
# stops/starts nothing itself (it pauses for the HA UI), so the "service needs
# a token" guard must only fire when the mode was actually asked for.
$AddonApiExplicit = $PSBoundParameters.ContainsKey('AddonApi')

$RepoRoot        = Split-Path -Parent $PSScriptRoot
$RepoOrchestrator = Join-Path $RepoRoot 'appdaemon\apps\battery_optimizer.py'
$RepoLib          = Join-Path $RepoRoot 'appdaemon\apps\battery_optimizer_lib'
$SmokeScript      = Join-Path $PSScriptRoot 'smoke_config.py'
$BackupNamePattern = '^backup-\d{8}-\d{6}$'
# Anything starting with backup- is treated as a stray backup that may be moved
# out of apps\ with -MoveStrayBackups (the old manual names are not timestamped).
$StrayBackupPattern = '^backup[-_.]'

# Backups must live OUTSIDE the apps directory: AppDaemon rglob()s app_dir for
# *.py and sys.path.insert(0, ...)s every __init__.py-less directory it finds
# one in, so a backup under apps\ shadows the package that was just deployed.
if ($BackupRoot -eq '') {
    $ShareRoot  = Split-Path -Parent $Share.TrimEnd('\')
    $BackupRoot = Join-Path (Join-Path $ShareRoot 'backups') 'battery_optimizer'
}
$LastDeployFile = Join-Path $BackupRoot 'last-deploy.txt'

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
# Add-on control (hassio.addon_* service / legacy HA Supervisor proxy) / pause
# ---------------------------------------------------------------------------
#
# On current HA (2026.8) the REST Supervisor proxy (/api/hassio/...) no longer
# forwards add-on info/start/stop for ANY token: GET /api/hassio/addons/<slug>
# /info answers HTTP 401 for an admin token too. The proxy forwards only a
# small allowlist - logs, ingress, backup upload/download - and add-on control
# lives on the admin-only websocket command supervisor/api, which we do not
# implement here. So -AddonApi proxy is LEGACY (older HA versions only) and the
# default -AddonApi service stops/starts the add-on with the hassio.addon_stop
# / hassio.addon_start services, which any authenticated user may call.
#
# The price is only that the add-on STATE cannot be queried: it comes from a
# TCP probe of AppDaemon's own HTTP port. The add-on LOG is on the proxy's
# allowlist and reads back fine with a non-admin token, so the post-deploy log
# check runs in both modes.

function Get-AddonHeaders {
    return @{ 'Authorization' = ('Bearer ' + $HaToken); 'Content-Type' = 'application/json' }
}

function Get-HaHostName {
    try {
        $parsed = [System.Uri]$HaUrl
        return [string]$parsed.Host
    } catch {
        return ''
    }
}

function Get-AddonApiDescription {
    if ($AddonApi -eq 'service') {
        return ("the hassio.addon_stop/addon_start services (POST " + $HaUrl.TrimEnd('/') +
                "/api/services/hassio/addon_<action>, any authenticated user's token), state probed on " +
                (Get-HaHostName) + ":" + $AppDaemonPort + " (AppDaemon's own HTTP server)")
    }
    return ("HA's Supervisor proxy (POST " + $HaUrl.TrimEnd('/') + "/api/hassio/addons/" + $AddonSlug +
            "/<action>, GET .../info to poll - LEGACY: current HA returns 401 there for every token)")
}

function Get-AddonState {
    $uri = ('{0}/api/hassio/addons/{1}/info' -f $HaUrl.TrimEnd('/'), $AddonSlug)
    $response = Invoke-RestMethod -Uri $uri -Method Get -Headers (Get-AddonHeaders) -TimeoutSec 30
    if ($null -eq $response) { return '' }
    if ($null -eq $response.data) { return '' }
    return [string]$response.data.state
}

function Test-AppDaemonPort {
    <#
      Is AppDaemon's HTTP server accepting connections? A plain TcpClient
      connect with a short timeout - quieter and much faster than
      Test-NetConnection, which does its own ping/DNS work and warns on
      failure (the expected outcome while we wait for a stop).
    #>
    param([int]$TimeoutMs = 3000)
    $hostName = Get-HaHostName
    if ($hostName -eq '') { return $false }
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect($hostName, $AppDaemonPort, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne($TimeoutMs, $false)) { return $false }
        $client.EndConnect($async)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

function Wait-AppDaemonPort {
    param([string]$Desired, [int]$TimeoutSeconds = 60)
    $hostName = Get-HaHostName
    if ($hostName -eq '') {
        Fail ("cannot derive a host name from -HaUrl '" + $HaUrl + "' for the AppDaemon port probe")
    }
    $wantOpen = ($Desired -eq 'started')
    $wanted = 'refuses connections'
    if ($wantOpen) { $wanted = 'accepts connections' }
    Write-Info ("state signal: TCP " + $hostName + ":" + $AppDaemonPort +
                " (-AddonApi service cannot query the add-on state); waiting until it " + $wanted)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $open = $false
    while ((Get-Date) -lt $deadline) {
        $open = Test-AppDaemonPort
        if ($open -eq $wantOpen) {
            Write-Ok ("add-on " + $Desired + " (port " + $AppDaemonPort + " " + $wanted + ")")
            return
        }
        $seen = 'closed'
        if ($open) { $seen = 'open' }
        Write-Info ("port " + $AppDaemonPort + " is " + $seen + ", waiting for '" + $Desired + "' ...")
        Start-Sleep -Seconds 3
    }
    $last = 'closed'
    if ($open) { $last = 'open' }
    Fail ("add-on did not reach state '" + $Desired + "' within " + $TimeoutSeconds +
          "s (port " + $AppDaemonPort + " last seen " + $last + ")")
}

function Invoke-AddonAction {
    param([ValidateSet('start', 'stop')][string]$Action)
    if ($AddonApi -eq 'service') {
        $uri = ('{0}/api/services/hassio/addon_{1}' -f $HaUrl.TrimEnd('/'), $Action)
        $body = @{ addon = $AddonSlug } | ConvertTo-Json -Compress
        Write-Info ("POST " + $uri + "  " + $body)
        Invoke-RestMethod -Uri $uri -Method Post -Headers (Get-AddonHeaders) -Body $body -TimeoutSec 120 | Out-Null
        return
    }
    $uri = ('{0}/api/hassio/addons/{1}/{2}' -f $HaUrl.TrimEnd('/'), $AddonSlug, $Action)
    Write-Info ("POST " + $uri)
    Invoke-RestMethod -Uri $uri -Method Post -Headers (Get-AddonHeaders) -TimeoutSec 120 | Out-Null
}

function Wait-AddonState {
    param([string]$Desired, [int]$TimeoutSeconds = 60)
    if ($AddonApi -eq 'service') {
        Wait-AppDaemonPort -Desired $Desired -TimeoutSeconds $TimeoutSeconds
        return
    }
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
        if ($HaToken -eq '') {
            Write-Plan ("pause and ask you to STOP add-on '" + $AddonSlug + "' in the HA UI (no -HaToken given)")
        } else {
            Write-Plan ("stop add-on '" + $AddonSlug + "' via " + (Get-AddonApiDescription))
        }
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
        if ($HaToken -eq '') {
            Write-Plan ("pause and ask you to START add-on '" + $AddonSlug + "' in the HA UI (no -HaToken given)")
        } else {
            Write-Plan ("start add-on '" + $AddonSlug + "' via " + (Get-AddonApiDescription))
        }
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
# Post-deploy health check (best effort - never fails the deploy)
# ---------------------------------------------------------------------------

function Get-AddonLogLines {
    <#
      GET /api/hassio/addons/<slug>/logs IS forwarded by the Supervisor proxy
      (it is on the allowlist that add-on info/start/stop is not) and answers
      200 for an admin and a non-admin long-lived token alike - so this check
      runs in both -AddonApi modes.

      The default window is only ~100 lines and the first full optimization
      dumps more than that immediately after startup, which scrolls the
      "Initializing Battery Optimizer" anchor out of view. The endpoint honours
      a journald-style `Range: entries=:-<N>:` header, but Windows PowerShell
      5.1 refuses to put `Range` (a .NET restricted header) on an
      Invoke-WebRequest ("must be modified using the appropriate property or
      method") and HttpWebRequest.AddRange only emits BYTE ranges. HttpClient's
      TryAddWithoutValidation is the way in; if anything about it fails we fall
      back to the plain call with the default window, because the whole
      post-deploy check is best effort.

      Journal lines carry ANSI colour codes - stripped here, before any anchor
      or pattern matching.
    #>
    param([int]$Lines = 400)
    $uri = ('{0}/api/hassio/addons/{1}/logs' -f $HaUrl.TrimEnd('/'), $AddonSlug)
    $text = ''
    $ranged = $false
    try {
        Add-Type -AssemblyName System.Net.Http
        $client = New-Object System.Net.Http.HttpClient
        try {
            $client.Timeout = [TimeSpan]::FromSeconds(30)
            $request = New-Object System.Net.Http.HttpRequestMessage -ArgumentList ([System.Net.Http.HttpMethod]::Get), $uri
            try {
                [void]$request.Headers.TryAddWithoutValidation('Authorization', ('Bearer ' + $HaToken))
                [void]$request.Headers.TryAddWithoutValidation('Accept', 'text/plain')
                [void]$request.Headers.TryAddWithoutValidation('Range', ('entries=:-' + $Lines + ':'))
                $response = $client.SendAsync($request).GetAwaiter().GetResult()
                try {
                    if (-not $response.IsSuccessStatusCode) {
                        throw ('HTTP ' + [int]$response.StatusCode + ' ' + $response.ReasonPhrase)
                    }
                    $text = [string]$response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
                    $ranged = $true
                } finally {
                    $response.Dispose()
                }
            } finally {
                $request.Dispose()
            }
        } finally {
            $client.Dispose()
        }
    } catch {
        $ranged = $false
    }
    if (-not $ranged) {
        $headers = @{ 'Authorization' = ('Bearer ' + $HaToken); 'Accept' = 'text/plain' }
        $response = Invoke-WebRequest -Uri ($uri + '?lines=' + $Lines) -Method Get -Headers $headers -TimeoutSec 30 -UseBasicParsing
        $text = [string]$response.Content
    }
    $esc = [char]27
    $text = $text -replace ($esc + '\[[0-9;?]*[A-Za-z]'), ''
    return @($text -split "`r?`n")
}

function Get-NewLogLines {
    <#
      Which lines are new since $Anchor? Timestamp formats differ between
      AppDaemon, the add-on wrapper and s6, so anchor on the literal last line
      captured before the start instead of parsing dates: find its LAST
      occurrence and return everything after it.
    #>
    param([string[]]$Lines, [string]$Anchor)
    if ($Anchor -eq '') { return @($Lines) }
    $index = -1
    for ($i = $Lines.Count - 1; $i -ge 0; $i--) {
        if ($Lines[$i] -eq $Anchor) { $index = $i; break }
    }
    if ($index -lt 0) { return @($Lines) }
    if ($index -ge ($Lines.Count - 1)) { return @() }
    return @($Lines[($index + 1)..($Lines.Count - 1)])
}

function Get-LogAnchor {
    param([int]$Lines = 200)
    try {
        $log = Get-AddonLogLines -Lines $Lines
    } catch {
        return ''
    }
    for ($i = $log.Count - 1; $i -ge 0; $i--) {
        if ($log[$i].Trim() -ne '') { return $log[$i] }
    }
    return ''
}

function Get-RepoAppVersion {
    # APP_VERSION = "x.y.z" in the orchestrator; the running app publishes the
    # same string as sensor.battery_optimizer's app_version attribute.
    try {
        $lines = @(Get-Content -LiteralPath $RepoOrchestrator)
    } catch {
        return ''
    }
    foreach ($line in $lines) {
        if ($line -match '^APP_VERSION\s*=\s*"([^"]+)"') { return $Matches[1] }
    }
    return ''
}

function Format-AttributeValue {
    param($Value)
    if ($null -eq $Value) { return '' }
    if ($Value -is [string]) { return $Value }
    if ($Value -is [System.Management.Automation.PSCustomObject]) {
        return ($Value | ConvertTo-Json -Compress -Depth 4)
    }
    if ($Value -is [System.Collections.IEnumerable]) {
        return ((@($Value) | ForEach-Object { [string]$_ }) -join ', ')
    }
    return [string]$Value
}

function Get-OptimizerAttribute {
    param([string]$Name)
    $uri = ('{0}/api/states/sensor.battery_optimizer' -f $HaUrl.TrimEnd('/'))
    $state = Invoke-RestMethod -Uri $uri -Method Get -Headers (Get-AddonHeaders) -TimeoutSec 30
    if ($null -eq $state) { return '' }
    $attrs = $state.attributes
    if ($null -eq $attrs) { return '' }
    $prop = $attrs.PSObject.Properties[$Name]
    if ($null -eq $prop) { return '' }
    return (Format-AttributeValue -Value $prop.Value)
}

function Invoke-RunningVersionCheck {
    <#
      The better half of the post-deploy check, and the one that catches a
      shadowing directory: the running app publishes APP_VERSION (and the paths
      it imported from) on sensor.battery_optimizer. /api/states is open to any
      authenticated user. Runs in both -AddonApi modes, alongside the log read.
      Best effort - it never fails the deploy.
    #>
    param([int]$TimeoutSeconds = 90)

    $expected = Get-RepoAppVersion
    if ($expected -eq '') {
        Write-Warn2 ("could not parse APP_VERSION from " + $RepoOrchestrator + " - skipping the running-version check.")
        return
    }
    Write-Info ("expecting sensor.battery_optimizer app_version = " + $expected)

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $seen = ''
    $lastFailure = ''
    while ((Get-Date) -lt $deadline) {
        try {
            $seen = Get-OptimizerAttribute -Name 'app_version'
            $lastFailure = ''
        } catch {
            $seen = ''
            $lastFailure = $_.Exception.Message
        }
        if ($seen -eq $expected) { break }
        Write-Info "waiting for sensor.battery_optimizer to report the new app_version ..."
        Start-Sleep -Seconds 5
    }

    if ($seen -eq $expected) {
        Write-Ok ("running app_version = " + $seen + " (matches the repo)")
        $paths = ''
        try {
            $paths = Get-OptimizerAttribute -Name 'code_paths'
        } catch {
            $paths = ''
        }
        if ($paths -ne '') { Write-Info ("code_paths: " + $paths) }
        return
    }

    if ($lastFailure -ne '') {
        Write-Warn2 ("could not read sensor.battery_optimizer (" + $lastFailure + ") - confirm the running version in the HA UI.")
        return
    }
    if ($seen -eq '') {
        Write-Warn2 ("sensor.battery_optimizer has no app_version attribute yet after " + $TimeoutSeconds +
                     "s. The app may just be slow to publish it, but if it stays empty check for a shadowing " +
                     "directory under apps\ or a stale __pycache__.")
        return
    }
    Write-Warn2 ("running app_version = " + $seen + " but the repo says " + $expected +
                 " - AppDaemon is NOT running the code that was just deployed (shadowing directory under apps\, " +
                 "stale __pycache__, or the add-on has not reloaded).")
}

function Invoke-PostDeployCheck {
    <#
      SHA256 verification proves the bytes on the share are right. It does NOT
      prove AppDaemon imported them - a shadowing directory (or a stale
      __pycache__) can make it run something else entirely. So read the log
      back (the log endpoint works with any token, in both -AddonApi modes) and
      confirm the version the app itself reports.
    #>
    param([string]$Anchor, [int]$TimeoutSeconds = 90)

    if ($HaToken -eq '') {
        Write-Warn2 "no -HaToken: cannot read the add-on log back. Check the HA UI (Settings > Add-ons > AppDaemon > Log) for 'Initializing Battery Optimizer' and any traceback."
        return
    }

    Invoke-AddonLogCheck -Anchor $Anchor -TimeoutSeconds $TimeoutSeconds
    Invoke-RunningVersionCheck -TimeoutSeconds $TimeoutSeconds
}

function Invoke-AddonLogCheck {
    param([string]$Anchor, [int]$TimeoutSeconds = 90)

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $initLine = ''
    $threadLine = ''
    $errorLines = @()
    $lastFailure = ''

    while ((Get-Date) -lt $deadline) {
        try {
            $log = Get-AddonLogLines -Lines 400
        } catch {
            $lastFailure = $_.Exception.Message
            Start-Sleep -Seconds 5
            continue
        }
        $lastFailure = ''
        $new = Get-NewLogLines -Lines $log -Anchor $Anchor

        foreach ($line in $new) {
            if ($line -match 'Initializing Battery Optimizer') { $initLine = $line }
            if ($line -match 'Starting apps with\s+(\d+)\s+worker threads?') { $threadLine = $line }
            if ($line -match 'ModuleNotFoundError|ImportError|Traceback \(most recent call last\)|Unexpected error loading|SyntaxError') {
                if ($errorLines -notcontains $line) { $errorLines += $line }
            }
        }
        if ($initLine -ne '' -or $errorLines.Count -gt 0) { break }
        Write-Info "waiting for 'Initializing Battery Optimizer' in the add-on log ..."
        Start-Sleep -Seconds 5
    }

    if ($lastFailure -ne '') {
        Write-Warn2 ("could not read the add-on log (" + $lastFailure + ") - check it manually in the HA UI")
        return
    }

    if ($threadLine -ne '') {
        Write-Info ("AppDaemon: " + $threadLine.Trim())
        if ($threadLine -match 'Starting apps with\s+(\d+)\s+worker threads?') {
            $threads = [int]$Matches[1]
            if ($threads -lt 2) {
                Write-Warn2 ("AppDaemon is running " + $threads + " worker thread(s). This app makes blocking " +
                             "set_wit_mode calls from callbacks; set appdaemon: total_threads: 4 in appdaemon.yaml.")
            }
        }
    }

    if ($errorLines.Count -gt 0) {
        Write-Host ''
        Write-Host ("    Import/startup errors in the add-on log after the restart:") -ForegroundColor Red
        foreach ($line in $errorLines) { Write-Host ("      " + $line.Trim()) -ForegroundColor Red }
        Write-Warn2 "the deploy finished but the app did NOT start cleanly - inspect the log and consider -Restore."
        return
    }

    if ($initLine -eq '') {
        Write-Warn2 ("no 'Initializing Battery Optimizer' line within " + $TimeoutSeconds +
                     "s. The app may just be slow, but if the log is silent check for a shadowing " +
                     "directory under apps\ or a stale __pycache__.")
        return
    }

    Write-Ok ("app initialized after the restart: " + $initLine.Trim())
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

function Test-PathInside {
    # $Candidate is $Root itself or below it (string compare; the share is UNC
    # and both sides come from the same parameters, so no resolution needed).
    param([string]$Root, [string]$Candidate)
    $r = $Root.TrimEnd('\')
    $c = $Candidate.TrimEnd('\')
    if ($c -ieq $r) { return $true }
    return $c.StartsWith($r + '\', [System.StringComparison]::OrdinalIgnoreCase)
}

function Set-DeployTimestamp {
    # Copy-Item preserves LastWriteTime, so a freshly deployed file keeps the
    # repo checkout's mtime and both AppDaemon's change detection and a human
    # `ls -l` on the share report a time that has nothing to do with the
    # deploy. Stamp them once, after the SHA256 verification.
    param([string[]]$Paths, [datetime]$When)
    $touched = 0
    foreach ($path in $Paths) {
        if (-not (Test-Path -LiteralPath $path)) { continue }
        try {
            (Get-Item -LiteralPath $path).LastWriteTime = $When
            $touched = $touched + 1
        } catch {
            Write-Warn2 ("could not stamp " + $path + ": " + $_.Exception.Message)
        }
    }
    return $touched
}

# ---------------------------------------------------------------------------
# Stray-app scan: nothing but our own files may hold *.py under apps\
# ---------------------------------------------------------------------------

function Get-StrayApps {
    <#
      AppDaemon 4.5 app discovery (app_management.py):
        - app_dir.rglob("*.py")                       -> every .py is a module
        - sys.path.insert(0, <dir>) for each directory containing .py files
          that has no __init__.py                     -> FRONT of sys.path
      So a directory under apps\ holding a copy of battery_optimizer.py or
      battery_optimizer_lib\ shadows the deployed one. Return every such path.
    #>
    param([string]$AppsDir, [string[]]$Allowed)

    $allowSet = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    [void]$allowSet.Add('battery_optimizer.py')
    foreach ($name in $Allowed) {
        if ($null -ne $name -and $name -ne '') { [void]$allowSet.Add($name) }
    }

    $strayFiles = @()
    $strayDirs = @()

    $topLevel = @(Get-ChildItem -Path $AppsDir -File -Filter '*.py' -ErrorAction SilentlyContinue)
    foreach ($file in $topLevel) {
        if (-not $allowSet.Contains($file.Name)) { $strayFiles += $file }
    }

    $subDirs = @(Get-ChildItem -Path $AppsDir -Directory -ErrorAction SilentlyContinue)
    foreach ($dir in $subDirs) {
        if ($dir.Name -ieq 'battery_optimizer_lib') { continue }
        if ($dir.Name -ieq '__pycache__') { continue }
        $py = @(Get-ChildItem -Path $dir.FullName -Recurse -File -Filter '*.py' -ErrorAction SilentlyContinue)
        if ($py.Count -gt 0) {
            $strayDirs += [pscustomobject]@{
                Directory = $dir
                PyCount   = $py.Count
                IsBackup  = ($dir.Name -match $StrayBackupPattern)
            }
        }
    }

    return [pscustomobject]@{ Files = @($strayFiles); Directories = @($strayDirs) }
}

function Move-StrayBackupDir {
    param([System.IO.DirectoryInfo]$Directory)
    $target = Join-Path $BackupRoot $Directory.Name
    if (Test-Path -LiteralPath $target) {
        $target = $target + '-moved-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
    }
    if ($DryRun) {
        Write-Plan ("move " + $Directory.FullName + " -> " + $target + " (never deleted)")
        return $target
    }
    if (-not (Test-Path -LiteralPath $BackupRoot -PathType Container)) {
        New-Item -ItemType Directory -Path $BackupRoot -Force | Out-Null
    }
    Move-Item -LiteralPath $Directory.FullName -Destination $target -Force
    Write-Ok ("moved out of apps\: " + $Directory.Name + " -> " + $target)
    return $target
}

function Invoke-StrayScan {
    <#
      Runs in -DryRun too: a shadowing directory makes the whole deploy a lie,
      so the rehearsal must surface it as loudly as the real run.
    #>
    param([string]$AppsDir, [switch]$Quiet)

    $stray = Get-StrayApps -AppsDir $AppsDir -Allowed $AllowedApps

    if ($stray.Files.Count -eq 0 -and $stray.Directories.Count -eq 0) {
        if (-not $Quiet) {
            Write-Ok ("nothing shadows the app: only battery_optimizer.py, battery_optimizer_lib\ and " +
                      ($AllowedApps -join ', ') + " hold *.py under apps\")
        }
        return
    }

    Write-Host ''
    Write-Host "    AppDaemon imports *.py from ANYWHERE under the apps directory." -ForegroundColor Yellow
    Write-Host "    Every directory with .py files and no __init__.py is prepended to sys.path," -ForegroundColor Yellow
    Write-Host "    so these shadow the files this script is about to deploy:" -ForegroundColor Yellow

    foreach ($file in $stray.Files) {
        Write-Host ("      FILE  " + $file.FullName) -ForegroundColor Red
    }
    foreach ($entry in $stray.Directories) {
        $tag = 'DIR '
        if ($entry.IsBackup) { $tag = 'BACKUP' }
        Write-Host ("      " + $tag + "  " + $entry.Directory.FullName +
                    "  (" + $entry.PyCount + " *.py)") -ForegroundColor Red
    }

    $backupDirs = @($stray.Directories | Where-Object { $_.IsBackup })
    $otherDirs  = @($stray.Directories | Where-Object { -not $_.IsBackup })

    if ($MoveStrayBackups -and $backupDirs.Count -gt 0) {
        Write-Host ''
        Write-Info "-MoveStrayBackups: evacuating backup directories (moved, never deleted)"
        foreach ($entry in $backupDirs) { Move-StrayBackupDir -Directory $entry.Directory | Out-Null }
    }

    $blocking = @()
    foreach ($file in $stray.Files) { $blocking += $file.FullName }
    foreach ($entry in $otherDirs) { $blocking += $entry.Directory.FullName }
    if (-not $MoveStrayBackups) {
        foreach ($entry in $backupDirs) { $blocking += $entry.Directory.FullName }
    }

    if ($blocking.Count -gt 0) {
        Write-Host ''
        Fail ("the apps directory contains " + $blocking.Count + " path(s) AppDaemon will import from, " +
              "shadowing the deployed code (this is how apps\backup-20260902-015911 made the add-on run " +
              "the PREVIOUS commit while SHA256 verification passed). Move or delete them by hand, " +
              "pass -MoveStrayBackups to relocate backup-* directories to " + $BackupRoot + ", " +
              "or allow a legitimate extra app with -AllowedApps.")
    }

    if (-not $DryRun) {
        $after = Get-StrayApps -AppsDir $AppsDir -Allowed $AllowedApps
        if ($after.Files.Count -gt 0 -or $after.Directories.Count -gt 0) {
            Fail "stray *.py paths still present under the apps directory after the move - aborting."
        }
        Write-Ok "apps directory is clean: nothing shadows battery_optimizer / battery_optimizer_lib"
    }
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
        # Accept a bare backup-<ts> name, resolved against the backup root.
        $candidate = Join-Path $BackupRoot $BackupDir
        if (Test-Path -LiteralPath $candidate -PathType Container) {
            Write-Info ("resolved '" + $BackupDir + "' to " + $candidate)
            $BackupDir = $candidate
        } else {
            Fail ("backup directory not found: " + $BackupDir)
        }
    }
    if (Test-PathInside -Root $Share -Candidate $BackupDir) {
        Write-Warn2 ("this backup lives INSIDE the apps directory (" + $BackupDir + ").")
        Write-Warn2 "AppDaemon imports *.py from there, so it has been shadowing the deployed code."
        Write-Warn2 ("it will be moved to " + $BackupRoot + " once the restore copy is done.")
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

    Write-Step "Evacuate any backup directory left inside apps\"
    # A rollback that leaves a backup-* directory under apps\ hands the add-on
    # a second copy of the package at the FRONT of sys.path, which is the bug
    # this whole rework exists to prevent.
    $strayAfter = Get-StrayApps -AppsDir $Share -Allowed $AllowedApps
    $backupDirsAfter = @($strayAfter.Directories | Where-Object { $_.IsBackup })
    $otherAfter = @($strayAfter.Directories | Where-Object { -not $_.IsBackup })
    if ($backupDirsAfter.Count -eq 0) {
        Write-Ok "no backup directory inside the apps directory"
    } else {
        foreach ($entry in $backupDirsAfter) { Move-StrayBackupDir -Directory $entry.Directory | Out-Null }
    }
    foreach ($entry in $otherAfter) {
        Write-Warn2 ("directory with *.py under apps\ that AppDaemon will import from: " + $entry.Directory.FullName)
    }
    foreach ($file in $strayAfter.Files) {
        Write-Warn2 ("unexpected *.py directly under apps\: " + $file.FullName)
    }

    Write-Step "Remove __pycache__ on the share"
    Remove-PyCache -Root $Share

    Write-Step "Start the add-on"
    $anchor = ''
    if (-not $DryRun -and $HaToken -ne '' -and -not $SkipPostCheck) { $anchor = Get-LogAnchor }
    Start-Addon

    if (-not $DryRun -and -not $SkipPostCheck) {
        Write-Step "Post-restore check (best effort)"
        Invoke-PostDeployCheck -Anchor $anchor -TimeoutSeconds $PostCheckTimeoutSeconds
    }

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
    if ($HaToken -eq '') {
        Write-Info ("addon : " + $AddonSlug + " - no -HaToken, so the script pauses for a manual stop/start" +
                    " (-AddonApi " + $AddonApi + " would use " + (Get-AddonApiDescription) + ")")
    } else {
        Write-Info ("addon : " + $AddonSlug + " via " + (Get-AddonApiDescription))
    }

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
    Write-Info ("backups: " + $BackupRoot)

    # -- 3. preflight: nothing under apps\ may shadow the deployed package ---
    Write-Step "Preflight: no shadowing *.py under the apps directory"
    Invoke-StrayScan -AppsDir $Share

    # -- 4. tests ------------------------------------------------------------
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

    # -- 5. compile ----------------------------------------------------------
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

    # -- 6. smoke test against the LIVE apps.yaml ----------------------------
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

    # -- 7. plan the copy ----------------------------------------------------
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

    # -- 8. backup -----------------------------------------------------------
    Write-Step "Backup the current share state (OUTSIDE the apps directory)"
    $timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $backupDir = Join-Path $BackupRoot ('backup-' + $timestamp)
    Write-Info ("backup root: " + $BackupRoot)
    $commitLines = @(
        ('deployed-commit : ' + $commit),
        ('short           : ' + $shortCommit),
        ('branch          : ' + $branch),
        ('deployed-at     : ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss K')),
        ('deployed-by     : ' + $env:USERNAME + '@' + $env:COMPUTERNAME),
        ('dirty-tree      : ' + ($dirty.Count -gt 0)),
        ('backup-of       : ' + $Share)
    )
    if ($DryRun) {
        Write-Plan ("create " + $backupDir + " with battery_optimizer.py + battery_optimizer_lib\ + deployed-commit.txt")
        Write-Plan ("write " + $LastDeployFile)
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
        Set-Content -LiteralPath (Join-Path $backupDir 'deployed-commit.txt') -Value $commitLines -Encoding utf8
        Write-Ok ("backup created: " + $backupDir)
    }

    # prune old backups
    $existingBackups = @(Get-ChildItem -Path $BackupRoot -Directory -ErrorAction SilentlyContinue |
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
        # -- 9. stop ---------------------------------------------------------
        Write-Step "Stop the AppDaemon add-on"
        Write-Info "AppDaemon hot-reloads on every .py write; a multi-file copy into a running add-on can import a new module against its old peers."
        Stop-Addon
        $stopped = $true

        # -- 10. copy ---------------------------------------------------------
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

        # -- 11. pycache -----------------------------------------------------
        Write-Step "Remove __pycache__ on the share"
        Remove-PyCache -Root $Share

        # -- 12. verify ------------------------------------------------------
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

        # -- 13. stamp the deploy time ---------------------------------------
        Write-Step "Stamp the deployed files with the deploy time"
        Write-Info "Copy-Item preserves the source LastWriteTime, which would make the share's mtimes describe the git checkout instead of the deploy."
        if ($DryRun) {
            Write-Plan ("set LastWriteTime = now on " + $plannedCopies.Count + " deployed file(s) and write " + $LastDeployFile)
        } else {
            $now = Get-Date
            $destinations = @()
            foreach ($item in $plannedCopies) { $destinations += $item.Destination }
            $touched = Set-DeployTimestamp -Paths $destinations -When $now
            Write-Ok ($touched.ToString() + " file(s) stamped " + $now.ToString('yyyy-MM-dd HH:mm:ss'))
            $lastDeployLines = @($commitLines + @(('backup          : ' + $backupDir)))
            Set-Content -LiteralPath $LastDeployFile -Value $lastDeployLines -Encoding utf8
            Write-Ok ("wrote " + $LastDeployFile)
        }

        # -- 14. start -------------------------------------------------------
        Write-Step "Start the AppDaemon add-on"
        $logAnchor = ''
        if (-not $DryRun -and $HaToken -ne '' -and -not $SkipPostCheck) { $logAnchor = Get-LogAnchor }
        Start-Addon

        # -- 15. post-deploy check (best effort) ------------------------------
        if (-not $DryRun -and -not $SkipPostCheck) {
            Write-Step "Post-deploy check: did the app actually start?"
            Invoke-PostDeployCheck -Anchor $logAnchor -TimeoutSeconds $PostCheckTimeoutSeconds
        } elseif ($DryRun) {
            Write-Step "Post-deploy check: did the app actually start?"
            Write-Plan ("poll the add-on log for up to " + $PostCheckTimeoutSeconds +
                        "s for 'Initializing Battery Optimizer', import errors and the worker-thread count")
            Write-Plan ("poll sensor.battery_optimizer until app_version = " + (Get-RepoAppVersion))
        }
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
            if ($AddonApi -ne 'service') { $restoreCmd = $restoreCmd + (' -AddonApi ' + $AddonApi) }
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
        Write-Host ''
        Write-Host 'SHA256 proves the bytes on the share; it does NOT prove AppDaemon imported them.' -ForegroundColor Yellow
        Write-Host 'Confirm the running code is the new one, e.g. sensor.battery_inverter_control_health' -ForegroundColor Yellow
        Write-Host 'must expose the attributes this commit adds.' -ForegroundColor Yellow
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

# Only a TYPED -AddonApi service without a token is an error. Falling into the
# default 'service' with no token is the plain manual-pause deploy, which must
# keep working exactly as before.
if ($AddonApiExplicit -and $AddonApi -eq 'service' -and $HaToken -eq '') {
    if ($DryRun) {
        # A rehearsal calls nothing, so let it run - but say plainly that the
        # same command without -DryRun would be refused.
        Write-Warn2 ("-AddonApi service without -HaToken: a REAL run would be refused here (calling " +
                     "hassio.addon_stop/addon_start needs an authenticated HA user). Continuing the rehearsal.")
    } else {
        Write-Host "-AddonApi service requires -HaToken: calling hassio.addon_stop/addon_start needs an authenticated" -ForegroundColor Red
        Write-Host "HA user (any user - no admin rights needed; `$env:HA_TOKEN is set on this machine). Or drop both" -ForegroundColor Red
        Write-Host "-AddonApi and -HaToken to stop/start the add-on by hand in the HA UI." -ForegroundColor Red
        exit 2
    }
}

if (Test-PathInside -Root $Share -Candidate $BackupRoot) {
    Write-Host ("-BackupRoot must NOT be inside the apps directory: " + $BackupRoot) -ForegroundColor Red
    Write-Host ("AppDaemon rglob()s " + $Share + " for *.py and prepends every __init__.py-less directory") -ForegroundColor Red
    Write-Host ("holding one to sys.path, so a backup there shadows the code you just deployed.") -ForegroundColor Red
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
