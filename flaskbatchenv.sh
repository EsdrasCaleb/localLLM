#!/bin/bash
#SBATCH --job-name=flask_chattester        # Job name
#SBATCH --output=flask_chattester_8b_%j.log    # Log file (%j = job ID)
#SBATCH --partition=amd-3tb           # Partition with GPU support (adjust as needed)
#SBATCH --time=12:00:00             # Test greather model in 12hours

# Load modules (adjust based on your environment)
#module load python/3.9              # Python version
module load cuda/11.7               # CUDA version (if using GPUs)

source $HOME/.bashrc
# Activate virtual environment (if needed)
conda activate llm_env

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
    echo "Successful execution of $folder/$env_file" >>executions.log
    echo "\nLog of $folder/$env_file:\n $output\n" >> logs.log
    rm $env_file
  else
    echo "Problem in execution of $folder/$env_file: $output" >>errors.log
  fi
  # After processing each project:
  end_time=$(date +%s)
  elapsed_time=$((end_time - start_time))
  echo "Processing $env_file took $elapsed_time seconds" >> timings.log
}

# Run main.py in the background
python3.9 main.py > flask_app.log 2>&1 &

flask_pid=$!
echo "Waiting for Flask app to initialize..."
while ! curl -s http://localhost:5000/list_models; do
  echo "Waiting for Flask app to be ready..."
  sleep 5
done


execute_command "$command" $1 "$folder"

kill $flask_pid

echo "All files processed. The system will exit now."
