# One password: zip ~/UAV on college machine, download, merge into this repo.
#
# Usage:
#   cd C:\code\UAV
#   .\scripts\pull_uav_from_college.ps1
#
# Smaller download (results only):
#   .\scripts\pull_uav_from_college.ps1 -ResultsOnly
#
# Skip re-zip if ~/uav_pull.zip already exists on server:
#   .\scripts\pull_uav_from_college.ps1 -ResultsOnly -DownloadOnly

param(
    [string]$RemoteUser = "cse2015",
    [string]$RemoteHost = "192.168.24.82",
    [switch]$ResultsOnly,
    [switch]$DownloadOnly
)

$ErrorActionPreference = "Stop"
$LocalRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$ZipName = "uav_pull.zip"
$LocalZip = Join-Path $LocalRoot $ZipName
$RemoteHome = "/home/$RemoteUser"
$RemoteZip = "$RemoteHome/$ZipName"

function Ensure-PoshSSH {
    if (Get-Module -ListAvailable -Name Posh-SSH) { return }
    Write-Host "Installing Posh-SSH for current user (no admin). One-time..."
    Set-PSRepository -Name PSGallery -InstallationPolicy Trusted -ErrorAction SilentlyContinue
    Install-Module -Name Posh-SSH -Scope CurrentUser -Force -AllowClobber
}

if ($ResultsOnly) {
    $ZipCmd = "cd $RemoteHome/UAV && rm -f $RemoteZip && zip -r $RemoteZip results -x 'results/archive/*' 'results/td3/*' 'results/*td3*' 'results/*.checkpoint.json' 'results/n*_campaign_*/*' && ls -lh $RemoteZip"
} else {
    $ZipCmd = "cd $RemoteHome && rm -f $RemoteZip && zip -r $RemoteZip UAV -x 'UAV/.git/*' 'UAV/**/__pycache__/*' 'UAV/**/venv/*' 'UAV/**/.venv/*' 'UAV/results/archive/*' 'UAV/results/td3/*' 'UAV/results/*.checkpoint.json' && ls -lh $RemoteZip"
}

function Invoke-ScpDownload {
    if (Test-Path $LocalZip) { Remove-Item -Force $LocalZip }
    scp "${RemoteUser}@${RemoteHost}:$RemoteZip" $LocalZip
    if ($LASTEXITCODE -ne 0) { throw "scp failed (exit $LASTEXITCODE)" }
}

function Invoke-PoshScpDownload {
    param($Credential)
    $item = Get-Command Get-SCPItem -ErrorAction SilentlyContinue
    $file = Get-Command Get-SCPFile -ErrorAction SilentlyContinue
    if ($item) {
        $p = $item.Parameters.Keys
        if ($p -contains "DestinationPath") {
            Get-SCPItem -ComputerName $RemoteHost -Credential $Credential -Path $RemoteZip -DestinationPath $LocalRoot -AcceptKey | Out-Null
            return
        }
        if ($p -contains "RemoteItem") {
            Get-SCPItem -ComputerName $RemoteHost -Credential $Credential -RemoteItem $RemoteZip -Path $LocalRoot -AcceptKey | Out-Null
            return
        }
    }
    if ($file) {
        Get-SCPFile -ComputerName $RemoteHost -Credential $Credential -RemoteFile $RemoteZip -LocalPath $LocalZip -AcceptKey | Out-Null
        return
    }
    throw "No Get-SCPItem/Get-SCPFile in Posh-SSH"
}

function Invoke-NativePull {
    param([switch]$DownloadOnly)
    if (-not $DownloadOnly) {
        Write-Host ""
        Write-Host "=== Step 1/2: SSH zip on server [password] ===" -ForegroundColor Cyan
        ssh "${RemoteUser}@${RemoteHost}" $ZipCmd
        if ($LASTEXITCODE -ne 0) { throw "remote zip failed (exit $LASTEXITCODE)" }
    }
    Write-Host ""
    Write-Host "=== Step 2/2: SCP download [password again if no keys] ===" -ForegroundColor Cyan
    Invoke-ScpDownload
}

$zipOnServer = $DownloadOnly
$session = $null
try {
    Ensure-PoshSSH
    Import-Module Posh-SSH -ErrorAction Stop
    Write-Host ""
    Write-Host "Enter SSH password for ${RemoteUser}@${RemoteHost} once:" -ForegroundColor Cyan
    $cred = Get-Credential -UserName $RemoteUser -Message "College SSH ($RemoteHost)"

    if (-not $DownloadOnly) {
        $session = New-SSHSession -ComputerName $RemoteHost -Credential $cred -AcceptKey
        Write-Host "Zipping on server (may take several minutes)..."
        $r = Invoke-SSHCommand -SessionId $session.SessionId -Command $ZipCmd -TimeOut 7200
        if ($r.ExitStatus -ne 0) {
            if ($r.Error) { Write-Host $r.Error }
            throw "remote zip failed: $($r.ExitStatus)"
        }
        if ($r.Output) { $r.Output | ForEach-Object { Write-Host $_ } }
        $zipOnServer = $true
    }

    if (Test-Path $LocalZip) { Remove-Item -Force $LocalZip }
    Write-Host "Downloading $RemoteZip ..."
    Invoke-PoshScpDownload -Credential $cred
    if ($session) {
        Remove-SSHSession -SessionId $session.SessionId | Out-Null
    }
} catch {
    Write-Warning "Posh-SSH path failed: $($_.Exception.Message)"
    if ($zipOnServer) {
        Write-Host "Zip is already on server; retrying download with scp only..." -ForegroundColor Yellow
        Invoke-NativePull -DownloadOnly
    } else {
        Write-Host "Falling back to ssh + scp..." -ForegroundColor Yellow
        Invoke-NativePull
    }
}

if (-not (Test-Path $LocalZip)) {
    throw "missing $LocalZip after download"
}

$mb = [math]::Round((Get-Item $LocalZip).Length / 1MB, 1)
Write-Host ""
Write-Host "Extracting ${mb} MB into $LocalRoot ..." -ForegroundColor Green

if ($ResultsOnly) {
    Expand-Archive -Path $LocalZip -DestinationPath $LocalRoot -Force
} else {
    $staging = Join-Path $env:TEMP "UAV_pull_extract"
    if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
    Expand-Archive -Path $LocalZip -DestinationPath $staging -Force
    $inner = Join-Path $staging "UAV"
    if (-not (Test-Path $inner)) { throw "expected UAV/ inside zip" }
    Write-Host "Merging into $LocalRoot ..."
    robocopy $inner $LocalRoot /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed $LASTEXITCODE" }
}

Write-Host "Done. Local repo: $LocalRoot" -ForegroundColor Green
Write-Host "You can delete $LocalZip to save space."
