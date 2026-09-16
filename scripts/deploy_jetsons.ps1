[CmdletBinding(SupportsShouldProcess)]
param(
    [string] $IdentityFile,
    [string] $Revision,
    [string] $RemoteDir = "~/hierarchical-fl-persistent",
    [string] $SshUserPrefix = "tachilab-orin"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (!$Revision) {
    $Revision = (& git -C $repoRoot rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or !$Revision) {
        throw "Could not resolve the local HEAD revision."
    }
}

$hosts = 201..207 | ForEach-Object { "192.168.10.$_" }
$sshOptions = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10")
if ($IdentityFile) { $sshOptions += @("-i", $IdentityFile) }

foreach ($ip in $hosts) {
    $nodeNumber = [int]$ip.Split('.')[-1] - 200
    $sshUser = "{0}{1:D2}" -f $SshUserPrefix, $nodeNumber
    $target = "$sshUser@$ip"
    if (!$PSCmdlet.ShouldProcess($target, "Install HFL revision $Revision")) { continue }

    & scp @sshOptions "$PSScriptRoot/remote_setup.sh" "${target}:/tmp/hfl_remote_setup.sh"
    if ($LASTEXITCODE -ne 0) { throw "scp failed: $target" }

    & ssh @sshOptions $target "bash /tmp/hfl_remote_setup.sh https://github.com/zoronosuke/hierarchical-fl.git $Revision $RemoteDir"
    if ($LASTEXITCODE -ne 0) { throw "remote setup failed: $target" }
}

Write-Host "Deployment complete on all seven Jetsons."
