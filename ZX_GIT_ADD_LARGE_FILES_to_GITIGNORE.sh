#!/bin/bash

# Set the size limit in bytes (100MB = 104857600 bytes)
sizeLimit=104857600

# Create a temporary file to store large files
tempFile=$(mktemp)

# Function to add large files to temp file
add_large_files() {
    local file="$1"
    if [[ -f "$file" ]]; then
        local size=$(stat -c%s "$file")
        if [[ $size -gt $sizeLimit ]]; then
            echo "${file#./}" >> "$tempFile"
        fi
    fi
}

# Export the function for use with 'find'
export -f add_large_files
export sizeLimit
export tempFile

# Loop through each file in the current directory and its subdirectories
find . -type f -exec bash -c 'add_large_files "$0"' {} \;

# Check and update .gitignore
if [[ ! -e .gitignore ]] || ! grep -q "# Large Files" .gitignore; then
    if [[ -s .gitignore ]] && [ "$(tail -c1 .gitignore)" != "" ]; then
        echo "" >> .gitignore
    fi
    echo -e "#--------------------------------------------------#\n# Large Files" >> .gitignore
fi

# Append large file paths to .gitignore
cat "$tempFile" >> .gitignore

# Remove duplicate lines from .gitignore
awk '!seen[$0]++' .gitignore > .gitignore.tmp
