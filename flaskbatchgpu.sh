#!/bin/bash
#SBATCH --job-name=flask_chattester_gpu        # Job name
#SBATCH --output=flask_gpu_%j.log    # Log file (%j = job ID)
#SBATCH --time=2-00:00:00            # Test greather model in 2 days
#SBATCH --nodes=1               # Use one node
#SBATCH --ntasks=4              # Run four tasks (processes)
#SBATCH --cpus-per-task=4       # Each task uses four CPU cores


# Load modules (adjust based on your environment)
#module load python/3.9              # Python version
module load libraries/cuda/12.6           # CUDA version (if using GPUs)

source $HOME/.bashrc
# Activate virtual environment (if needed)
conda activate llm_env_gpu
pip install --upgrade torch torchvision torchaudio

# Function to execute a command and capture its output
execute_command() {
  local command="$1"
  local env_file="$2"
  local folder="$3"
  start_time=$(date +%s)
  echo "Executing: $command"
  local output=$(eval "$command" 2>&1)
  local exit_code=$?

  if [ $exit_code -eq 0 ]; then
    echo "Successful execution of $folder/$env_file" >>executions_gpu.log
    echo "\nLog of $folder/$env_file:\n $output\n" >> logs_gpu.log
    rm $env_file
  else
    echo "Problem in execution of $folder/$env_file: $output" >>errors_gpu.log
  fi
  # After processing each project:
  end_time=$(date +%s)
  elapsed_time=$((end_time - start_time))
  echo "Processing $env_file took $elapsed_time seconds" >> timings_gpu.log
}

# Run main.py in the background
python3.9 main.py >> flask_app_gpu.log 2>&1 &

flask_pid=$!
echo "Waiting for Flask app to initialize..."
while ! curl -s http://localhost:5000/health; do
  echo "Waiting for Flask app to be ready..."
  sleep 5
done


# Loop through each folder in the "envs" directory
for folder in $(ls -d enfiles/* | sort -r); do
#for folder in enfiles/*; do
  if [ -d "$folder" ]; then
    case "$folder" in
            *gg)
                continue ;; # Skip folders ending in "gg"
    esac
    echo "Processing folder: $folder"

    # Loop through each .env file in the folder
    for env_file in "$folder"/*; do
      # Construct the command
      command="java -jar chatunitest-standalone.jar $env_file project"
      # Execute the command and capture output
      execute_command "$command" "$env_file" "$folder"
    done
    echo "Clear Models"
    python3.9 clear_models.py
  fi
done

# Stop Flask app
#kill $flask_pid

echo "All files processed. The system will exit now."
