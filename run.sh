#!/usr/bin/env bash
# Works on Mac and Linux (Ubuntu)
set -e
cd "$(dirname "$0")"

if ! command -v python3 &>/dev/null; then
  echo "Python 3 is required. Install it with: sudo apt install python3 python3-pip  (Ubuntu)"
  echo "  or via https://python.org (Mac)"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q -r requirements.txt

echo ""
echo "Starting AI Label Studio at http://localhost:5000"
echo "Press Ctrl+C to stop."
echo ""
python app.py
