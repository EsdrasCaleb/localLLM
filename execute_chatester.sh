execute_command() {
  local command="$1"
  local env_file="$2"
  local folder="$3"

  echo "Executing: $command"
  local output=$(eval "$command" 2>&1)
  local exit_code=$?

  if [ $exit_code -eq 0 ]; then
    sudo -u caleb echo "Successful execution of $folder/$env_file" >>executions.log 
    sudo -u calebca echo "\nLog of $folder/$env_file:\n $output\n" >> logs.log
    rm $env_file
  else
    echo "Problem in execution of $folder/$env_file: $output" >>errors.log
  fi
}
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=""
# Run main.py in the background
python main.py &
flask_pid=$!
# Wait for Flask to initialize (use sleep or health check)
echo "Waiting for Flask app to initialize..."
sleep 15  # Adjust this as needed
# Loop through each folder in the "envs" directory
for folder in enfiles/*; do
  if [ -d "$folder" ]; then
    echo "Processing folder: $folder"

    # Loop through each .env file in the folder
    for env_file in "$folder"/*; do
      # Construct the command
      command="sudo -u caleb java -jar chatunitest-standalone-1.0.0.jar $env_file project"
      testcommand="sudo -u caleb java -jar chatunitest-standalone-1.0.0.jar $env_file test $env_file"
      # Execute the command and capture output
      execute_command "$testcommand" "$env_file" "$folder"
    done
    echo "Clear Models"
    python clear_models.py
  fi
done
# Shutdown the computer
kill $flask_pid
echo "All files processed. The system will shut down now."
shutdown -h now
