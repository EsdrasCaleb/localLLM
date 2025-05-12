#!/bin/bash
#SBATCH --job-name=flask_uni        # Job name
#SBATCH --output=flask_uni_%j.log    # Log file (%j = job ID)
#SBATCH --time=2-00:00:00            # Test greater model in 2 days

# Load modules (adjust based on your environment)
#module load python/3.10              # Python version
module load libraries/cuda/12.6              # CUDA version (if using GPUs)
module load cmake
source $HOME/.bashrc
# Activate virtual environment (if needed)
conda activate llm_env_gpu
#conda install gcc_linux-64 libstdcxx-ng cmake ninja
#conda install -c conda-forge cmake make gcc libgcc gxx -y
#pip install --upgrade -r requirements.txt
#pip install --no-cache-dir llama-cpp-python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

rm -r /tmp/chatunitest-info/firebird
# Function to execute a command and capture its output
execute_command() {
  local command="$1"
  local env_file="$2"
  local folder="$3"
  start_time=$(date +%s)
  echo "$(date '+%Y-%m-%d %H:%M:%S') Executing: $command"
  local output=$(eval "$command" 2>&1)
  local exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo "Successful execution of $env_file" >> "unilogs/executions_$folder.log"
    echo "\nLog of $env_file:\n $output\n" >> "unilogs/logs_$folder.log"
    #rm $env_file
  else
    echo "Problem in execution of $env_file: $output" >>"unilogs/errors_$folder.log"
  fi
  # After processing each project:
  end_time=$(date +%s)
  elapsed_time=$((end_time - start_time))
  echo "Processing $env_file took $elapsed_time seconds" >> timings.log
}

# Run main.py in the background
python3.9 main.py >> "flask_app_gpu_$(date '+%Y-%m-%d_%H_%M_%S').log" 2>&1 &

echo "Waiting for Flask app to initialize..."
while ! curl -s http://localhost:5000/health; do
  echo "Waiting for Flask app to be ready..."
  sleep 5
done

# Loop through all folders passed as arguments
for folder in "$@"; do
  if [ -d "$folder" ]; then
    last_folder=$(basename "$folder")

    # Loop through each environment file in the current folder and execute the command
    for env_file in "$folder"/*; do
      if [ -f "$env_file" ]; then
        command="java -jar chatunitest-standalone.jar $env_file project"
        execute_command "$command" "$env_file" "$last_folder"
      fi
    done
    echo "Clear Models"
    python3.9 clear_models.py
  else
    echo "Directory $folder does not exist. Skipping..."
  fi
done

#rm -r ../scratch/models
echo "All files processed. The system will exit now."
