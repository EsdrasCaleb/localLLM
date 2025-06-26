#!/bin/bash
#SBATCH --job-name=flask_uni        # Job name
#SBATCH --output=log_flask_uni_%j.log    # Log file (%j = job ID)
#SBATCH --time=2-00:00:00            # tempo maximo no a100
#SBATCH --gres=gpu:2                 # Request 1 GPU
#SBATCH --qos=preempt


# Load modules (adjust based on your environment)
#module load python/3.10              # Python version
module load libraries/cuda/12.6              # CUDA version (if using GPUs)
module load cmake
source ~/.bashrc
source $HOME/.bashrc
# Activate virtual environment (if needed)
conda activate llm_env_gpu
#conda install gcc_linux-64 libstdcxx-ng cmake ninja
#conda install -c conda-forge cmake make gcc libgcc gxx -y
#pip install --upgrade -r requirements.txt
#pip install --no-cache-dir llama-cpp-python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Function to execute a command and capture its output

execute_command() {
  local command="$1"
  local env_file="$2"
  start_time=$(date +%s)
  echo "$(date '+%Y-%m-%d %H:%M:%S') Executing: $command"
  local output=$(eval "$command" 2>&1)
  local exit_code=$?
  local filename=$(basename "$env_file")  # Extracts only the filename
  local timestamp=$(date +"%Y%m%d_%H%M%S")  # Generates a timestamp
  if [ $exit_code -eq 0 ]; then
    echo "Successful execution of $env_file" >> "unilogs/executions_$filename_$timestamp.log"
    echo "\nLog of $env_file:\n $output\n" >> "unilogs/logs_$filename_$timestamp.log"
    #rm $env_file
  else
    echo "Problem in execution of $env_file: $output" >>"unilogs/errors_$filename_$timestamp.log"
  fi
  # After processing each project:
  end_time=$(date +%s)
  elapsed_time=$((end_time - start_time))
  echo "Processing $env_file took $elapsed_time seconds" >> timings.log
}
# Run main.py in the background
python main.py >> flask_app_gpu.log 2>&1 &

echo "Waiting for Flask app to initialize..."
while ! curl -s http://localhost:5000/health; do
  echo "Waiting for Flask app to be ready..."
  sleep 5
done


if [ -d executed ]; then
  echo "Directory executed already exists."
else
  echo "Directory executed does not exist. Creating..."
  mkdir executed
  echo "Directory executed created."
fi

# Loop through each folder in the "envs" directory
#for folder in $(ls -d enfiles/* | sort -r); do
for folder in enfiles/*; do
  if [ -d "$folder" ]; then
    #case "$folder" in
    #        *gg)
    #            continue ;; # Skip folders ending in "gg"
    #esac
    echo "Processing folder: $folder"

    # Loop through each .env file in the folder
    for env_file in "$folder"/*; do
      # Construct the command
      command="java -jar chatunitest-standalone.jar $env_file project"
      # Execute the command and capture output
      execute_command "$command" "$env_file"
      # Cria o diretório de destino, preservando estrutura
      target_dir="executed/$(basename "$folder")"
      mkdir -p "$target_dir"

      # Move o arquivo
      mv "$env_file" "$target_dir/"
      echo "Moved file $env_file to $target_dir/"
      echo "Readed file $env_file"
    done
    echo "Clear Models"
    python clear_models.py
  fi
done

# Stop Flask app
#kill $flask_pid

echo "All files processed. The system will exit now."
