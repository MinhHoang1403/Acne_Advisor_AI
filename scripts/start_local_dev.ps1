param(
    [int]$ApiPort = 8000,
    [int]$FrontendPort = 5173,
    [string]$ApiHost = "127.0.0.1",
    [ValidateRange(10, 600)]
    [int]$StartupTimeoutSeconds = 240,
    [switch]$DryRun,
    [switch]$SkipFrontend
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$backendUrl = "http://$ApiHost`:$ApiPort"
$frontendUrl = "http://localhost:$FrontendPort"
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$frontendDir = Join-Path $repoRoot "src\frontend"
$logDir = Join-Path $repoRoot "logs\local-dev"
$viteCli = Join-Path $frontendDir "node_modules\vite\bin\vite.js"

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
        [int]$TimeoutSeconds = $StartupTimeoutSeconds
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        $health = Test-BackendHealth -BaseUrl $BaseUrl
        if ($health.Reachable -and $health.Response.service -eq "acne-advisor-api") {
            return $health
        }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)

    throw "Backend did not respond at $BaseUrl/health within $TimeoutSeconds seconds. See $logDir\backend.stderr.log."
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

if (-not (Test-Path (Join-Path $repoRoot ".env"))) {
    throw "Missing .env. Copy .env.example to .env and configure the required provider credentials."
}

$pythonVersion = & $pythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0 -or $pythonVersion -ne "3.11") {
    throw "The project venv must use Python 3.11. Recreate it with 'py -3.11 -m venv .venv'."
}

if (-not $SkipFrontend) {
    $nodeCommand = Get-Command node.exe -ErrorAction Stop
    if (-not (Test-Path $viteCli)) {
        throw "Frontend dependencies are missing. Run 'npm.cmd ci' in $frontendDir."
    }
}

if (-not $DryRun) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

Write-Step "Checking Docker services."
if ($DryRun) {
    Write-Step "DryRun: would run 'docker compose up -d --pull never --no-build postgres redis qdrant neo4j'."
} else {
    Push-Location $repoRoot
    try {
        docker compose up -d --pull never --no-build postgres redis qdrant neo4j
        if ($LASTEXITCODE -ne 0) {
            throw "Docker services failed to start. Check Docker Desktop and the pinned datastore images."
        }
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
    if ($health.Reachable -and $health.Response.service -eq "acne-advisor-api") {
        Write-Step "Existing backend is reachable with health status '$($health.Status)'. Reusing it."
    } else {
        throw "Port $ApiPort is occupied but $backendUrl/health is not reachable. Stop the owning process manually if it is safe. This script will not kill unknown processes."
    }
} else {
    if ($DryRun) {
        Write-Step "DryRun: would start backend on $backendUrl."
    } else {
        Write-Step "Starting backend."
        $backendProcess = Start-Process `
            -FilePath $pythonExe `
            -ArgumentList @("-m", "uvicorn", "src.api.app:app", "--reload", "--reload-dir", "src", "--host", $ApiHost, "--port", "$ApiPort") `
            -WorkingDirectory $repoRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $logDir "backend.stdout.log") `
            -RedirectStandardError (Join-Path $logDir "backend.stderr.log") `
            -PassThru
        $backendProcess.Id | Set-Content (Join-Path $logDir "backend.pid")
        $health = Wait-BackendHealth -BaseUrl $backendUrl
        Write-Step "Backend health status: $($health.Status)"
    }
}

if ($SkipFrontend) {
    Write-Step "SkipFrontend enabled. Not starting Vite."
} elseif ($DryRun) {
    Write-Step "DryRun: would run 'npm.cmd run dev -- --host 127.0.0.1 --port $FrontendPort' in $frontendDir."
} elseif (Get-PortListener -Port $FrontendPort) {
    $frontendListener = Get-PortListener -Port $FrontendPort
    $statePath = Join-Path $logDir "frontend.json"
    $existing = if (Test-Path $statePath) { Get-Content $statePath -Raw | ConvertFrom-Json } else { $null }
    $runningFrontend = Get-CimInstance Win32_Process -Filter "ProcessId=$($frontendListener.OwningProcess)"
    $sameVite = $runningFrontend -and $runningFrontend.CommandLine -and
        $runningFrontend.CommandLine.IndexOf($viteCli, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
    if ($sameVite -and $existing -and $existing.pid -eq $frontendListener.OwningProcess -and $existing.apiUrl -eq $backendUrl) {
        Write-Step "Reusing the frontend already started for $backendUrl."
    } else {
        throw "Frontend port $FrontendPort is occupied. Stop that frontend or choose another port and update CORS_ALLOW_ORIGINS."
    }
} else {
    Write-Step "Starting frontend."
    $previousApiUrl = $env:VITE_API_URL
    try {
        $env:VITE_API_URL = $backendUrl
        $frontendProcess = Start-Process `
            -FilePath $nodeCommand.Source `
            -ArgumentList @(('"{0}"' -f $viteCli), "--host", "127.0.0.1", "--port", "$FrontendPort", "--strictPort") `
            -WorkingDirectory $frontendDir `
            -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $logDir "frontend.stdout.log") `
            -RedirectStandardError (Join-Path $logDir "frontend.stderr.log") `
            -PassThru
        @{ pid = $frontendProcess.Id; apiUrl = $backendUrl } | ConvertTo-Json |
            Set-Content (Join-Path $logDir "frontend.json")
        $frontendDeadline = (Get-Date).AddSeconds(30)
        do {
            $frontendProcess.Refresh()
            $frontendListener = Get-PortListener -Port $FrontendPort
            if ($frontendProcess.HasExited -or ($frontendListener -and $frontendListener.OwningProcess -eq $frontendProcess.Id)) {
                break
            }
            Start-Sleep -Milliseconds 250
        } while ((Get-Date) -lt $frontendDeadline)
        if ($frontendProcess.HasExited -or -not $frontendListener -or $frontendListener.OwningProcess -ne $frontendProcess.Id) {
            throw "Frontend failed to start. See $logDir\frontend.stderr.log."
        }
    } finally {
        if ($null -eq $previousApiUrl) {
            Remove-Item Env:VITE_API_URL -ErrorAction SilentlyContinue
        } else {
            $env:VITE_API_URL = $previousApiUrl
        }
    }
}

Write-Step "Swagger: $backendUrl/docs"
Write-Step "Health: $backendUrl/health"
Write-Step "Frontend: $frontendUrl"
Write-Step "Done."
