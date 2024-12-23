    #!/bin/bash
    #SBATCH --job-name=flask_8b         # Job name
    #SBATCH --output=flask_8b_%j.log    # Log file (%j = job ID)
    #SBATCH --ntasks=1                  # Number of tasks (single-node app)
    #SBATCH --cpus-per-task=8           # CPUs for model inference
    #SBATCH --mem=32G                   # Memory allocation
    #SBATCH --gpus=1                    # Request 1 GPU (if available)
    #SBATCH --time=4:00:00              # Max runtime
    #SBATCH --partition=gpu             # Partition with GPU support (adjust as needed)

    # Load modules (adjust based on your environment)
    module load python/3.8              # Python version
    module load cuda/11.7               # CUDA version (if using GPUs)
    source $HOME/.bashrc
    # Activate virtual environment (if needed)
    source vllm_env/bin/activate  # Update with your virtual environment path

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
    echo "\nLog of $folder/$env_file:\n $output\n" >> succes.log
  else
    echo "Problem in execution of $folder/$env_file: $output" >>result.log
  fi
}


# Run main.py in the background
python main.py &
# Loop through each folder in the "envs" directory
for folder in enfiles/*; do
  if [ -d "$folder" ]; then
    echo "Processing folder: $folder"

    # Loop through each .env file in the folder
    for env_file in "$folder"/*; do
      # Construct the command
      command="java -jar ../chatunitest-standalone/target/chatunitest-standalone-1.0.0.jar $env_file project"
      testcommand="java -jar ../chatunitest-standalone/target/chatunitest-standalone-1.0.0.jar $env_file test $env_file"
      # Execute the command and capture output
      execute_command "$command" "$env_file" "$folder"
    done
    echo "Clear Models"
    python clear_models.py
  fi
done
# Shutdown the computer
echo "All files processed. The system will shut down now."
sudo shutdown -h now
