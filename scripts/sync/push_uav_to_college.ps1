# One password: zip this repo locally, upload, unzip into ~/UAV on college machine.
#
# Usage:
#   cd C:\code\UAV
#   .\scripts\push_uav_to_college.ps1

param(
    [string]$RemoteUser = "cse2015",
    [string]$RemoteHost = "192.168.24.82",
    [string]$RemoteUavDir = "UAV"
)

$ErrorActionPreference = "Stop"
$LocalRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$ZipName = "uav_sync.zip"
$ZipPath = Join-Path $env:TEMP $ZipName
$Staging = Join-Path $env:TEMP "UAV_sync_staging"
$RemoteHome = "/home/$RemoteUser"
$RemoteZip = "$RemoteHome/$ZipName"
$RemoteDest = "$RemoteHome/$RemoteUavDir"

function Ensure-PoshSSH {
    if (Get-Module -ListAvailable -Name Posh-SSH) { return }
    Write-Host "Installing Posh-SSH for current user (no admin). One-time..."
    Set-PSRepository -Name PSGallery -InstallationPolicy Trusted -ErrorAction SilentlyContinue
    Install-Module -Name Posh-SSH -Scope CurrentUser -Force -AllowClobber
}

function Build-Staging {
    if (Test-Path $Staging) { Remove-Item -Recurse -Force $Staging }
    New-Item -ItemType Directory -Force -Path $Staging | Out-Null

    foreach ($d in @("src", "scripts", "data", "docs", "latex", "tests")) {
        $src = Join-Path $LocalRoot $d
        if (Test-Path $src) {
            Copy-Item -Recurse -Force $src (Join-Path $Staging $d)
        }
    }

    foreach ($f in @("pyproject.toml", "requirements.txt", "README.md")) {
        $p = Join-Path $LocalRoot $f
        if (Test-Path $p) { Copy-Item -Force $p $Staging }
    }

    $resDest = Join-Path $Staging "results"
    New-Item -ItemType Directory -Force -Path $resDest | Out-Null
    $resLocal = Join-Path $LocalRoot "results"

    Get-ChildItem $resLocal -File -Filter "campaign_8.8mhz_cap25*.json" -ErrorAction SilentlyContinue |
        Copy-Item -Destination $resDest -Force
    Get-ChildItem $resLocal -File -Filter "campaign_sca_anchor_*" -ErrorAction SilentlyContinue |
        Copy-Item -Destination $resDest -Force

    foreach ($sub in @("n100", "n100_500m_cap25", "n200")) {
        $p = Join-Path $resLocal $sub
        if (Test-Path $p) { Copy-Item -Recurse -Force $p (Join-Path $resDest $sub) }
    }

    foreach ($pat in @("n20_campaign_*", "n100_campaign_*", "n200_campaign_*")) {
        Get-ChildItem $resLocal -Directory -Filter $pat -ErrorAction SilentlyContinue |
            ForEach-Object {
                Copy-Item -Recurse -Force $_.FullName (Join-Path $resDest $_.Name)
            }
    }
}

Write-Host "Building staging folder..."
Build-Staging

if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
Write-Host "Zipping to $ZipPath ..."
Compress-Archive -Path (Join-Path $Staging "*") -DestinationPath $ZipPath -CompressionLevel Fastest
$zipMb = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
Write-Host "Local zip: ${zipMb} MB"

$unzipCmd = "mkdir -p $RemoteDest && unzip -o $RemoteZip -d $RemoteDest && rm -f $RemoteZip && echo OK"

try {
    Ensure-PoshSSH
    Import-Module Posh-SSH -ErrorAction Stop
    Write-Host ""
    Write-Host "Enter SSH password for ${RemoteUser}@${RemoteHost} (once):" -ForegroundColor Cyan
    $cred = Get-Credential -UserName $RemoteUser -Message "College SSH ($RemoteHost)"

    Write-Host "Uploading..."
    Set-SCPItem -ComputerName $RemoteHost -Credential $cred -Path $ZipPath -Destination $RemoteHome -AcceptKey

    $session = New-SSHSession -ComputerName $RemoteHost -Credential $cred -AcceptKey
    Write-Host "Unzipping on server into $RemoteDest ..."
    $r = Invoke-SSHCommand -SessionId $session.SessionId -Command $unzipCmd -TimeOut 3600
    if ($r.ExitStatus -ne 0) {
        Write-Host $r.Error
        throw "remote unzip failed: $($r.ExitStatus)"
    }
    if ($r.Output) { $r.Output | ForEach-Object { Write-Host $_ } }
    Remove-SSHSession -SessionId $session.SessionId | Out-Null
} catch {
    Write-Warning "Posh-SSH failed: $($_.Exception.Message)"
    Write-Host "Fallback: scp + ssh (two password prompts)..." -ForegroundColor Yellow
    scp $ZipPath "${RemoteUser}@${RemoteHost}:~/${ZipName}"
    if ($LASTEXITCODE -ne 0) { throw "scp failed" }
    ssh "${RemoteUser}@${RemoteHost}" $unzipCmd
    if ($LASTEXITCODE -ne 0) { throw "remote unzip failed" }
}

Write-Host ""
Write-Host "Done. Remote tree: $RemoteDest" -ForegroundColor Green
Write-Host "Next on server: source ~/venvs/uavdt/bin/activate && cd ~/UAV && ./scripts/remote/college_gpu_nohup.sh"
