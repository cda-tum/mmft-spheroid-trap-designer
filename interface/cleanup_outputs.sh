#!/bin/bash

# To execute on server please run:
# 0 3 * * * bash /var/www/cda/app/mmftspheroidtrap/interface/cleanup_outputs.sh >> /var/log/cleanup_outputs.log 2>&1


# Directory containing the generated output folders
TARGET_DIR="$(dirname "$0")/static/outputs"

# Delete folders older than 2 days
find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d -mtime +2 -exec rm -rf {} \;