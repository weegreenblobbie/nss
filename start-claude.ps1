<#
.SYNOPSIS
Launches the claude-agent Docker container and starts the claude CLI.

Previously build: docker build -t nss-claude .
#>

Write-Host "Starting Claude Agent container..." -ForegroundColor Cyan

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
    nss-claude