#!/bin/bash

timestamp() {
    # It's helpful to add a newline character or return the value for cleaner use
    # return is for exit codes, use echo to return a value to the caller
    echo $(date +"%Y-%m-%d_%H:%M:%S") 
}

# Fix 1: Ensure this variable is defined if you use it right away, 
# or move the echo command after the function call below.
LOG_TIMESTAMP=$(timestamp)
echo "Script started at $LOG_TIMESTAMP" >> script.log

prompt_with_default() {
    local prompt_message=$1
    local default_value=$2
    read -p "$prompt_message (default: $default_value): " user_input
    echo "${user_input:-$default_value}"
}

# --- Main Script ---

echo "🚀 Initializing a Next.js 16 project with pnpm, App Router, TypeScript, and Tailwind CSS"

if ! command -v pnpm &> /dev/null; then # Improved check syntax
    echo "pnpm is not installed. Please install pnpm first (e.g., using 'npm install -g pnpm')."
    exit 1    
fi

# Fix 2: Use command substitution $(timestamp) to execute the function
PROJECT_NAME=$(prompt_with_default "Enter your project name" "fws_nextjs_base_$(timestamp)")

echo "Creating project: $PROJECT_NAME..."

pnpm exec create-next-app@latest "$PROJECT_NAME" --yes

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "🎉 Successfully created Next.js project '$PROJECT_NAME' using pnpm."
    echo "To get started:"
    echo "cd $PROJECT_NAME"
    echo "pnpm run dev"
else
    echo "❌ Failed to create Next.js project."
    exit 1
fi