[CmdletBinding(SupportsShouldProcess)]
param(
    [string] $PublicKeyFile = "$HOME/.ssh/id_ed25519.pub",
    [string] $SshUserPrefix = "tachilab-orin"
)

$ErrorActionPreference = "Stop"
if (!(Test-Path -LiteralPath $PublicKeyFile)) {
    throw "Public key not found: $PublicKeyFile"
}

$publicKey = (Get-Content -Raw -LiteralPath $PublicKeyFile).Trim()
$encodedPublicKey = [Convert]::ToBase64String(
    [Text.Encoding]::ASCII.GetBytes($publicKey + "`n")
)
$hosts = 201..207 | ForEach-Object { "192.168.10.$_" }

foreach ($ip in $hosts) {
    $nodeNumber = [int]$ip.Split('.')[-1] - 200
    $sshUser = "{0}{1:D2}" -f $SshUserPrefix, $nodeNumber
    $target = "$sshUser@$ip"
    if (!$PSCmdlet.ShouldProcess($target, "Install SSH public key")) { continue }

    Write-Host "Registering key on $target (enter this node's password when prompted)..."
    # Base64 avoids PowerShell/native-pipeline encoding and whitespace splitting.
    $remoteCommand = "umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; echo $encodedPublicKey | base64 -d >> ~/.ssh/authorized_keys; sort -u ~/.ssh/authorized_keys -o ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys"
    & ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new $target $remoteCommand
    if ($LASTEXITCODE -ne 0) { throw "Key registration failed: $target" }

    & ssh -o BatchMode=yes -o ConnectTimeout=10 $target "true"
    if ($LASTEXITCODE -ne 0) {
        throw "Key was copied but key authentication still failed: $target"
    }
    Write-Host "Verified key authentication on $target"
}

Write-Host "SSH public key registered on all seven Jetsons."
