param([int]$HttpsPort = 8443)
$ErrorActionPreference = 'Stop'
if ($HttpsPort -lt 1 -or $HttpsPort -gt 65535) { throw 'Invalid HTTPS port' }
$rawStatus = & tailscale status --json
if ($LASTEXITCODE -ne 0) { throw 'Could not query Tailscale' }
$status = $rawStatus | ConvertFrom-Json
if ($status.BackendState -ne 'Running') { throw 'Connect Tailscale first.' }
$rawConfig = & tailscale serve status --json
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect existing Serve configuration' }
$config = $rawConfig | ConvertFrom-Json
$dns = $status.Self.DNSName.TrimEnd('.')
$hostPort = "${dns}:$HttpsPort"
$target = 'http://127.0.0.1:7860'
$webEntry = if ($config.Web) { $config.Web.PSObject.Properties[$hostPort] } else { $null }
if ($webEntry) {
    $handlers = @($webEntry.Value.Handlers.PSObject.Properties)
    if ($handlers.Count -ne 1 -or $handlers.Name -ne '/' -or $handlers.Value.Proxy -ne $target) {
        throw "Port $HttpsPort already serves another app. Choose a different -HttpsPort."
    }
} elseif ($config.TCP -and $config.TCP.PSObject.Properties["$HttpsPort"]) {
    throw "Port $HttpsPort already has a TCP service. Choose a different -HttpsPort."
}
& tailscale serve --bg "--https=$HttpsPort" $target
if ($LASTEXITCODE -ne 0) { throw 'Tailscale Serve configuration failed' }
Write-Output "Browser: https://$hostPort/"
Write-Output "REST docs: https://$hostPort/docs"
