[CmdletBinding(SupportsShouldProcess)]
param(
    [string] $IdentityFile,
    [string] $RemoteDir = "~/hierarchical-fl-persistent",
    [string] $SshUserPrefix = "tachilab-orin",
    [string] $RunId = (Get-Date -Format "yyyyMMdd-HHmmss"),
    [int] $StartupTimeoutSeconds = 180,
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
if ($RunId -notmatch '^[A-Za-z0-9._-]+$') {
    throw "RunId may contain only letters, digits, dot, underscore, and hyphen."
}

$sshOptions = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10")
if ($IdentityFile) { $sshOptions += @("-i", $IdentityFile) }
$dryRunArg = if ($DryRun) { " --dry-run" } else { "" }
$localAddresses = "localhost,127.0.0.1,192.168.10.201,192.168.10.202,192.168.10.203,192.168.10.204,192.168.10.205,192.168.10.206,192.168.10.207"
$jetsonEnv = "env LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/libcudss/12:/usr/local/cuda/lib64 NO_PROXY=$localAddresses no_proxy=$localAddresses no_grpc_proxy=$localAddresses"

function Get-SshTarget([string] $Ip) {
    $nodeNumber = [int]$Ip.Split('.')[-1] - 200
    $sshUser = "{0}{1:D2}" -f $SshUserPrefix, $nodeNumber
    return "$sshUser@$Ip"
}

function Start-Role([string] $Ip, [string] $Command) {
    $target = Get-SshTarget $Ip
    $logName = "hfl-$($Ip.Split('.')[-1]).log"
    $remoteCommand = "cd $RemoteDir && mkdir -p logs/$RunId && setsid -f $jetsonEnv $Command > logs/$RunId/$logName 2>&1 < /dev/null"
    if ($PSCmdlet.ShouldProcess($target, $Command)) {
        # setsid -f detaches remotely, so SSH exits before the next node starts.
        & ssh @sshOptions $target $remoteCommand
        if ($LASTEXITCODE -ne 0) { throw "start failed: $target" }
        return $true
    }
    return $false
}

function Wait-RemoteListener([string] $Ip, [int] $Port) {
    $target = Get-SshTarget $Ip
    $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        & ssh @sshOptions $target "ss -ltn | grep -q ':$Port '" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Verified listener $Ip`:$Port"
            return
        }
        Start-Sleep -Seconds 2
    }
    throw "Timed out waiting for $Ip`:$Port. Check $RemoteDir/logs/$RunId/."
}

$cfg = "config/jetson-5tier"
$common = "--global-config $cfg/global.yaml --topology-config $cfg/topology.yaml --defaults-config config/defaults.yaml$dryRunArg"
$globalCommand = ".venv/bin/python -m src.core.global_server --config $cfg/global.yaml$dryRunArg"
function Edge-Command([string] $Id) { ".venv/bin/python -m src.core.run_edge --edge-config $cfg/$Id.yaml $common" }
function Leaf-Command([string] $Id, [string] $Parent, [int] $Partition) {
    ".venv/bin/python -m src.core.run_leaf --client-id $Id --edge-address $Parent --partition-id $Partition $common"
}

# 5段構成: Global → Edge 01/02 → L01(中継) → L02 → L03(中継) → L04
# 中継ノードは親の待受確認後に起動し、自身の待受確認後に子を起動する。
$globalStarted = Start-Role "192.168.10.201" $globalCommand
if ($globalStarted) { Wait-RemoteListener "192.168.10.201" 8080 }

$edge01Started = Start-Role "192.168.10.202" (Edge-Command "edge_01")
$edge02Started = Start-Role "192.168.10.203" (Edge-Command "edge_02")
if ($edge01Started) { Wait-RemoteListener "192.168.10.202" 9001 }
if ($edge02Started) { Wait-RemoteListener "192.168.10.203" 9002 }

$leaf01Started = Start-Role "192.168.10.204" (Edge-Command "leaf_01")
Start-Role "192.168.10.205" (Leaf-Command "leaf_02" "192.168.10.202:9001" 2) | Out-Null
if ($leaf01Started) { Wait-RemoteListener "192.168.10.204" 9003 }

$leaf03Started = Start-Role "192.168.10.206" (Edge-Command "leaf_03")
if ($leaf03Started) { Wait-RemoteListener "192.168.10.206" 9004 }

Start-Role "192.168.10.207" (Leaf-Command "leaf_04" "192.168.10.206:9004" 5) | Out-Null

Write-Host "All 5-tier roles launched. Run ID: $RunId"
Write-Host "Logs: $RemoteDir/logs/$RunId/ on each Jetson."
