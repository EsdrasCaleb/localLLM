#!/bin/bash

# Function to execute a command and capture its output
execute_command() {
  local command="$1"
  local env_file="$2"
  local folder="$3"

  echo "Executing: $command"
  local output=$(eval "$command" 2>&1)
  local exit_code=$?

  if [ $exit_code -eq 0 ]; then
    echo "Successful execution of $folder/$env_file"
  else
    echo "Problem in execution of $folder/$env_file: $output"
  fi
}

# Loop through each folder in the "envs" directory
for folder in envs/*; do
  if [ -d "$folder" ]; then
    echo "Processing folder: $folder"

    # Loop through each .env file in the folder
    for env_file in "$folder"/*.env; do
      # Construct the command
      command="java -jar target/chatunitest-standalone-1.0.0.jar $env_file project"

      # Execute the command and capture output
      execute_command "$command" "$env_file" "$folder"
    done
  fi
done