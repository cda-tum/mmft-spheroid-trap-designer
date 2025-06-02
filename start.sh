#!/bin/bash

# Absolute path to your repo (adjust if needed)
APP_DIR="/var/www/cda/app/mmft-spheroid-trap-designer"

echo "[DEBUG $(date +'%Y-%m-%dT%H:%M:%S')] start.sh wurde aufgerufen" >> /tmp/deploy-debug.log

# Activate virtual environment
source "$APP_DIR/trap-designer/bin/activate"

# Move into the repo directory
cd "$APP_DIR"

# Start Gunicorn
gunicorn --preload --worker-class=gevent --worker-connections=1000 --workers 4 -t 1000 --bind 127.0.0.1:5002 'app:start_server()'
