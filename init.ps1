# MedQA Multi-Agent System — Init Script (PowerShell)
# Run with: .\init.ps1

$ErrorActionPreference = "Stop"
$ProgressPreference     = "SilentlyContinue"

$ROOT_DIR = $PSScriptRoot
if (-not $ROOT_DIR) { $ROOT_DIR = Get-Location }
Set-Location $ROOT_DIR

# ── Commands ─────────────────────────────────────────────────────────────────
$INSTALL_CMD = @("uv", "sync", "--all-groups")
$VERIFY_CMD  = @("uv", "run", "pytest")
$START_CMD   = @("uv", "run", "python", "-m", "main")

# ── [1/5] Working directory ───────────────────────────────────────────────────
Write-Host "==> [1/5] Working directory: $PWD"

# ── [2/5] Syncing dependencies ───────────────────────────────────────────────
Write-Host "==> [2/5] Syncing dependencies"
& @($INSTALL_CMD[0]) $INSTALL_CMD[1..($INSTALL_CMD.Length - 1)]
if ($LASTEXITCODE -ne 0) { exit 1 }

# ── [3/5] Verifying harness files ────────────────────────────────────────────
Write-Host "==> [3/5] Verifying harness files..."
$FILES_OK = $true
foreach ($file in @("AGENTS.md", "CLAUDE.md", "feature_list.json", "clean-state-checklist.md")) {
    if (-not (Test-Path $file -PathType Leaf)) {
        Write-Host "  MISSING: $file"
        $FILES_OK = $false
    } else {
        Write-Host "  OK: $file"
    }
}

if ($FILES_OK -ne $true) {
    Write-Host "=== Init complete with warnings. Some harness files are missing. ==="
    exit 1
}

# ── [4/5] Running baseline verification ──────────────────────────────────────
Write-Host "==> [4/5] Running baseline verification"
$VERIFY_CMD_STR = ($VERIFY_CMD -join " ")
Write-Host "  Running: $VERIFY_CMD_STR"
Invoke-Expression $VERIFY_CMD_STR
$PYTEST_EXIT = $LASTEXITCODE

# ── [5/5] Startup command ─────────────────────────────────────────────────────
Write-Host "==> [5/5] Startup command"
$START_CMD_STR = ($START_CMD | ForEach-Object { "'$_'" }) -join " "
Write-Host "  $START_CMD_STR"

# ── Optional: launch the app ───────────────────────────────────────────────────
if ($env:RUN_START_COMMAND -eq "1") {
    Write-Host "==> Starting the app"
    Invoke-Expression $START_CMD_STR
}

Write-Host ""
Write-Host "Set `$env:RUN_START_COMMAND=1 if you want init.ps1 to launch the app directly."

if ($PYTEST_EXIT -ne 0) {
    Write-Host "WARNING: pytest exited with code $PYTEST_EXIT"
    exit $PYTEST_EXIT
}

exit 0
