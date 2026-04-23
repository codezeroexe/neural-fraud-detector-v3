#!/bin/bash
# Neural Fraud Detector v2 - Double-click to run
# Place this file next to launch.py and double-click to run

cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv and run
echo "Starting Neural Fraud Detector v2..."
source venv/bin/activate
python launch.py