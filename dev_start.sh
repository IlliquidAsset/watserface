#!/bin/bash
set -e

echo "=== Claude Code Dev Environment Setup ==="
echo "Timestamp: $(date)"

# Install Node.js if not present
if ! command -v node &> /dev/null; then
    echo "Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
    apt-get update && apt-get install -y nodejs
fi

# Install code-server if not present
if ! command -v code-server &> /dev/null; then
    echo "Installing code-server..."
    curl -fsSL https://code-server.dev/install.sh | sh
fi

# Install Claude Code CLI via npm
echo "Installing Claude Code CLI..."
if ! command -v claude-code &> /dev/null; then
    npm install -g @anthropic/claude-code
fi

# Authentication setup
echo ""
echo "=== CLAUDE CODE AUTHENTICATION REQUIRED ==="
echo "Please run the following command to authenticate:"
echo "claude-code auth login"
echo ""
echo "Your Organization ID: 7d37921e-6314-4b53-a02d-7ea9040b3afb"
echo ""
echo "After authentication, Claude Code will be available in VS Code"
echo "=============================================="
echo ""

# Create session state directory
mkdir -p /home/user/app/.claude-session

# Log session start
echo "$(date): Dev environment started" >> /home/user/app/.claude-session/audit.log
echo "Node.js: $(node --version)" >> /home/user/app/.claude-session/audit.log
echo "npm: $(npm --version)" >> /home/user/app/.claude-session/audit.log
echo "code-server: $(code-server --version | head -1)" >> /home/user/app/.claude-session/audit.log

# Start code-server
echo "Starting VS Code server on port 8080..."
exec code-server \
    --bind-addr 0.0.0.0:8080 \
    --auth none \
    --disable-telemetry \
    --disable-update-check \
    --install-extension ms-python.python \
    /home/user/app
