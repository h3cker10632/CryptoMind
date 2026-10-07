# CryptoMind supervisor (Windows) — keeps the server alive across restarts & crashes.
# - "Restart server" button: process exits, this loop relaunches it (picking up code).
# - "Kill server" button: writes .shutdown marker, loop exits for real.
# - Crashes: relaunched after 2s (state.json restores everything).
Set-Location $PSScriptRoot
Remove-Item -Force .shutdown -ErrorAction SilentlyContinue
$env:CRYPTOMIND_SUPERVISED = "1"
Write-Host "[supervisor] starting CryptoMind"
while ($true) {
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
    $code = $LASTEXITCODE
    if (Test-Path .shutdown) {
        Remove-Item -Force .shutdown -ErrorAction SilentlyContinue
        Write-Host "[supervisor] shutdown requested — exiting for real (code $code)"
        break
    }
    Write-Host "[supervisor] server exited (code $code) — restarting in 2s"
    Start-Sleep -Seconds 2
}

