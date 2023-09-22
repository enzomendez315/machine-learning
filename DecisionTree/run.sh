#!/bin/bash

# Grant executable permission if needed
if [ ! -x "$0" ]; then
  chmod +x "$0"
fi

# Get the path to the directory containing the script
script_dir="$(cd "$(dirname "$0")" && pwd)"

# # Specify the relative path to your files
# files_dir="relative/path/to/files"

# # Construct the full path to the files
# files_path="$script_dir/$files_dir"

# # Check if the directory exists
# if [ ! -d "$files_path" ]; then
#     echo "Files directory does not exist: $files_path"
# fi

# Run Python scripts
echo "Running car_decision_tree.py"
python3 car_decision_tree.py

echo "Running bank_decision_tree.py"
python3 bank_decision_tree.py

#echo "Scripts completed."

# Use this command to make the script an executable
# chmod +x myscript.sh