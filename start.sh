#!/bin/bash

# Absolute path to your repo (adjust if needed)
APP_DIR="/var/www/cda/app/mmft-spheroid-trap-designer"

if [ ! -d "$APP_DIR/trap-designer" ]; then
  python3 -m venv "$APP_DIR/trap-designer"
  source "$APP_DIR/trap-designer/bin/activate"
  pip install --upgrade pip
  pip install -r "$APP_DIR/requirements.txt"
else
  source "$APP_DIR/trap-designer/bin/activate"
fi

# Activate virtual environment
source "$APP_DIR/trap-designer/bin/activate"

# Move into the repo directory
cd "$APP_DIR"

# Start Gunicorn
gunicorn --preload --worker-class=gevent --worker-connections=1000 --workers 4 -t 1000 --bind 127.0.0.1:5002 'app:start_server()'
