#!/bin/bash
#SBATCH --job-name=flask_uni        # Job name
#SBATCH --output=flask_uni_%j.log    # Log file (%j = job ID)
#SBATCH --partition=amd-3tb           # Partition with GPU support (adjust as needed)
#SBATCH --time=20:00:00             # Test greather model in 12hours
#SBATCH --nodes=1               # Use one node
#SBATCH --ntasks=4              # Run four tasks (processes)
#SBATCH --cpus-per-task=4       # Each task uses four CPU cores


# Load modules (adjust based on your environment)
#module load python/3.9              # Python version
#module load libraries/cuda               # CUDA version (if using GPUs)

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
    echo "Successful execution of $folder/$env_file" >> "unilogs/executions_$file_sigle.log"
    echo "\nLog of $folder/$env_file:\n $output\n" >> "unilogs/logs_$file_sigle.log"
    rm $env_file
  else
    echo "Problem in execution of $folder/$env_file: $output" >>"unilogs/errors_$file_single.log"
  fi
  # After processing each project:
  end_time=$(date +%s)
  elapsed_time=$((end_time - start_time))
  echo "Processing $env_file took $elapsed_time seconds" >> timings.log
}

file=$(basename "$1")
# Run main.py in the background
python3.9 main.py >> "unilogs/flask_app_$file.log" 2>&1 &

echo "Waiting for Flask app to initialize..."
while ! curl -s http://localhost:5000/health; do
  echo "Waiting for Flask app to be ready..."
  sleep 5
done


execute_command "$command" $1 $2


echo "All files processed. The system will exit now."
