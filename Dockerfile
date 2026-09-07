# Start with a lightweight Node.js image
FROM node:20-slim

# Install git (often used by Gemini CLI for repo analysis)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install the official Google Gemini CLI globally
RUN npm install -g @google/gemini-cli

# Set the working directory
WORKDIR /workspace

# Keep the container alive
CMD ["/bin/bash"]
