param(
    [int]$ApiPort = 8000,
    [int]$FrontendPort = 5173,
    [string]$ApiHost = "127.0.0.1",
    [switch]$DryRun,
    [switch]$SkipFrontend
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$backendUrl = "http://$ApiHost`:$ApiPort"
$frontendUrl = "http://localhost:$FrontendPort"
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$frontendDir = Join-Path $repoRoot "src\frontend"

function Write-Step {
    param([string]$Message)
    Write-Host "[local-dev] $Message"
}

function Get-PortListener {
    param([int]$Port)
    return Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
}

function Get-ProcessInfo {
    param([int]$ProcessId)
    return Get-Process -Id $ProcessId -ErrorAction SilentlyContinue |
        Select-Object Id,ProcessName,Path,StartTime
}

function Test-BackendHealth {
    param([string]$BaseUrl)
    try {
        $response = Invoke-RestMethod -Method Get -Uri "$BaseUrl/health" -TimeoutSec 5
        return [pscustomobject]@{
            Reachable = $true
            Status = $response.status
            Response = $response
        }
    } catch {
        return [pscustomobject]@{
            Reachable = $false
            Status = "unreachable"
            Error = $_.Exception.Message
        }
    }
}

function Wait-BackendHealth {
    param(
        [string]$BaseUrl,
        [int]$TimeoutSeconds = 45
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        $health = Test-BackendHealth -BaseUrl $BaseUrl
        if ($health.Reachable) {
            return $health
        }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)

    throw "Backend did not respond at $BaseUrl/health within $TimeoutSeconds seconds."
}

Write-Step "Repository: $repoRoot"
Write-Step "Backend URL: $backendUrl"
Write-Step "Frontend URL: $frontendUrl"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker CLI not found. Start Docker Desktop or install Docker CLI."
}

if (-not (Test-Path $pythonExe)) {
    throw "Python venv not found: $pythonExe"
}

if (-not (Test-Path $frontendDir)) {
    throw "Frontend directory not found: $frontendDir"
}

Write-Step "Checking Docker services."
if ($DryRun) {
    Write-Step "DryRun: would run 'docker compose up -d'."
} else {
    Push-Location $repoRoot
    try {
        docker compose up -d
    } finally {
        Pop-Location
    }
}

$listener = Get-PortListener -Port $ApiPort
if ($listener) {
    $processInfo = Get-ProcessInfo -ProcessId $listener.OwningProcess
    Write-Step "Port $ApiPort is already in use."
    $processInfo | Format-List | Out-String | Write-Host

    $health = Test-BackendHealth -BaseUrl $backendUrl
    if ($health.Reachable) {
        Write-Step "Existing backend is reachable with health status '$($health.Status)'. Reusing it."
    } else {
        throw "Port $ApiPort is occupied but $backendUrl/health is not reachable. Stop the owning process manually if it is safe. This script will not kill unknown processes."
    }
} else {
    if ($DryRun) {
        Write-Step "DryRun: would start backend on $backendUrl."
    } else {
        Write-Step "Starting backend."
        Start-Process `
            -FilePath $pythonExe `
            -ArgumentList @("-m", "uvicorn", "src.api.app:app", "--reload", "--host", $ApiHost, "--port", "$ApiPort") `
            -WorkingDirectory $repoRoot `
            -WindowStyle Normal
        $health = Wait-BackendHealth -BaseUrl $backendUrl
        Write-Step "Backend health status: $($health.Status)"
    }
}

if ($SkipFrontend) {
    Write-Step "SkipFrontend enabled. Not starting Vite."
} elseif ($DryRun) {
    Write-Step "DryRun: would run 'npm.cmd run dev -- --host 127.0.0.1 --port $FrontendPort' in $frontendDir."
} else {
    Write-Step "Starting frontend."
    Start-Process `
        -FilePath "npm.cmd" `
        -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1", "--port", "$FrontendPort") `
        -WorkingDirectory $frontendDir `
        -WindowStyle Normal
}

Write-Step "Swagger: $backendUrl/docs"
Write-Step "Health: $backendUrl/health"
Write-Step "Frontend: $frontendUrl"
Write-Step "Done."
