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

$globalCommand = ".venv/bin/python -m src.core.global_server --config config/jetson-7node/global.yaml$dryRunArg"
$edge01Command = ".venv/bin/python -m src.core.run_edge --edge-config config/jetson-7node/edge_01.yaml --global-config config/jetson-7node/global.yaml --topology-config config/topology.yaml --defaults-config config/defaults.yaml$dryRunArg"
$edge02Command = ".venv/bin/python -m src.core.run_edge --edge-config config/jetson-7node/edge_02.yaml --global-config config/jetson-7node/global.yaml --topology-config config/topology.yaml --defaults-config config/defaults.yaml$dryRunArg"
$leafCommands = [ordered]@{
    "192.168.10.204" = ".venv/bin/python -m src.core.run_leaf --client-id leaf_01 --edge-address 192.168.10.202:9001 --partition-id 1 --global-config config/jetson-7node/global.yaml --topology-config config/topology.yaml --defaults-config config/defaults.yaml$dryRunArg"
    "192.168.10.205" = ".venv/bin/python -m src.core.run_leaf --client-id leaf_02 --edge-address 192.168.10.202:9001 --partition-id 2 --global-config config/jetson-7node/global.yaml --topology-config config/topology.yaml --defaults-config config/defaults.yaml$dryRunArg"
    "192.168.10.206" = ".venv/bin/python -m src.core.run_leaf --client-id leaf_03 --edge-address 192.168.10.203:9002 --partition-id 4 --global-config config/jetson-7node/global.yaml --topology-config config/topology.yaml --defaults-config config/defaults.yaml$dryRunArg"
    "192.168.10.207" = ".venv/bin/python -m src.core.run_leaf --client-id leaf_04 --edge-address 192.168.10.203:9002 --partition-id 5 --global-config config/jetson-7node/global.yaml --topology-config config/topology.yaml --defaults-config config/defaults.yaml$dryRunArg"
}

$globalStarted = Start-Role "192.168.10.201" $globalCommand
if ($globalStarted) { Wait-RemoteListener "192.168.10.201" 8080 }

$edge01Started = Start-Role "192.168.10.202" $edge01Command
$edge02Started = Start-Role "192.168.10.203" $edge02Command
if ($edge01Started) { Wait-RemoteListener "192.168.10.202" 9001 }
if ($edge02Started) { Wait-RemoteListener "192.168.10.203" 9002 }

foreach ($entry in $leafCommands.GetEnumerator()) {
    Start-Role $entry.Key $entry.Value | Out-Null
}

Write-Host "All roles launched. Run ID: $RunId"
Write-Host "Logs: $RemoteDir/logs/$RunId/ on each Jetson."
