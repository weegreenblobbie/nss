<#
.SYNOPSIS
Launches the claude-agent Docker container and starts the claude CLI.

.PARAMETER Resume
Optional session UUID. When given, claude starts with `--resume <uuid>`.

.EXAMPLE
.\start-claude.ps1 f4786be0-1a35-49f8-b031-da4c414def42

Previously build: docker build -t nss-claude .
#>
param(
    [Parameter(Position = 0)]
    [string]$Resume
)

Write-Host "Starting Claude Agent container..." -ForegroundColor Cyan

# Arguments after the image name are appended to the Dockerfile ENTRYPOINT
$claudeArgs = if ($Resume) { @("--resume", $Resume) } else { @() }

# Run the docker container
# -it keeps it interactive
# --rm removes the container when you exit
# -v mounts your current Windows folder to /workspace
# 
docker run -it --rm `
    -v "${PWD}:/workspace" `
    -v "${PWD}/claude-cfg:/home/ubuntu/.claude" `
    -e TERM=xterm-256color `
    -e COLORTERM=truecolor `
    nss-claude @claudeArgs