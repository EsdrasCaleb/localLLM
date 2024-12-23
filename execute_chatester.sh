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
    echo "Successful execution of $folder/$env_file" >>result.log
  else
    echo "Problem in execution of $folder/$env_file: $output" >>result.log
  fi
}
# Activate the virtual environment
source vllm_env/bin/activate

# Run main.py in the background
python main.py &
# Loop through each folder in the "envs" directory
for folder in enfiles/*; do
  if [ -d "$folder" ]; then
    echo "Processing folder: $folder"

    # Loop through each .env file in the folder
    for env_file in "$folder"/*; do
      # Construct the command
      command="java -jar target/chatunitest-standalone-1.0.0.jar $env_file project"
      testcommand="java -jar target/chatunitest-standalone-1.0.0.jar $env_file test $env_file"
      # Execute the command and capture output
      execute_command "$testcommand" "$env_file" "$folder"
    done
    python request.py 
  fi
done
# Shutdown the computer
echo "All files processed. The system will shut down now."
sudo shutdown -h now