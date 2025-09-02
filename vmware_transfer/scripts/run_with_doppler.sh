#!/bin/bash
# Run any Python script with Doppler secrets injected

if [ -z "$1" ]; then
    echo "Usage: ./scripts/run_with_doppler.sh <python_script> [args...]"
    exit 1
fi

# Run with Doppler using full path
C:/tools/doppler/doppler.exe run -- python "$@"

#Make It Executable

chmod +x scripts/run_with_doppler.sh