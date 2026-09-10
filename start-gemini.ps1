<#
.SYNOPSIS
Launches the gemini-agent Docker container and starts the Gemini CLI.
#>

# 1. Check if the API key is already in the environment
$apiKey = $env:GEMINI_API_KEY

# 2. If it's missing, prompt the user to enter it securely
if ([string]::IsNullOrWhiteSpace($apiKey)) {
    Write-Host "GEMINI_API_KEY not found in environment." -ForegroundColor Yellow
    $apiKey = Read-Host "Please enter your Gemini API Key"
    
    if ([string]::IsNullOrWhiteSpace($apiKey)) {
        Write-Error "API Key is required to run the Gemini CLI."
        exit 1
    }
} else {
    Write-Host "Found GEMINI_API_KEY in environment." -ForegroundColor Green
}

Write-Host "Starting Gemini Agent container..." -ForegroundColor Cyan

# 3. Run the docker container
# -it keeps it interactive
# --rm removes the container when you exit
# -v mounts your current Windows folder to /workspace
# -e forwards the API key and terminal color flags to eliminate the 256-color warning
# The final 'gemini' command tells the container to launch the CLI directly
docker run -it --rm `
    -v "${PWD}:/workspace" `
    -e GEMINI_API_KEY=$apiKey `
    -e TERM=xterm-256color `
    -e COLORTERM=truecolor `
    gemini-agent gemini