<#
.SYNOPSIS
Launches the gemini-agent Docker container and starts the Gemini CLI.
#>

Write-Host "Starting Google Antigravity Agent container..." -ForegroundColor Cyan

# Run the docker container
# -it keeps it interactive
# --rm removes the container when you exit
# -v mounts your current Windows folder to /workspace
# 
docker run -it --rm `
    -v "${PWD}:/workspace" `
    -e TERM=xterm-256color `
    -e COLORTERM=truecolor `
    antigravity-cli 
