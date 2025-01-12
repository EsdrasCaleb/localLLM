#!/bin/bash
#SBATCH --job-name=flask_uni        # Job name
#SBATCH --output=flask_uni_%j.log    # Log file (%j = job ID)
#SBATCH --time=2-00:00:00            # Test greather model in 2 days


# Load modules (adjust based on your environment)
#module load python/3.10              # Python version
module load libraries/cuda/12.6              # CUDA version (if using GPUs)
module load cmake
source $HOME/.bashrc
# Activate virtual environment (if needed)
conda activate llm_env_gpu
conda install gcc_linux-64 libstdcxx-ng cmake ninja
conda install -c conda-forge cmake make gcc libgcc gxx llama-cpp conda-forge::llama-cpp-python conda-forge::llama.cpp -y
pip install --upgrade -r requirements.txt
pip install --no-cache-dir llama-cpp-python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Function to execute a command and capture its output
execute_command() {
  local command="$1"
  local env_file="$2"
  local file="$3"
  start_time=$(date +%s)
  echo "Executing: $command"
  local output=$(eval "$command" 2>&1)
  local exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo "Successful execution of $env_file" >> "unilogs/executions_$file.log"
    echo "\nLog of $env_file:\n $output\n" >> "unilogs/logs_$file.log"
    #rm $env_file
  else
    echo "Problem in execution of $env_file: $output" >>"unilogs/errors_$file.log"
  fi
  # After processing each project:
  end_time=$(date +%s)
  elapsed_time=$((end_time - start_time))
  echo "Processing $env_file took $elapsed_time seconds" >> timings.log
}

#file=$(basename "$1")
last_folder=$(basename "$(dirname "$1")")
# Run main.py in the background
python3.9 main.py >> "unilogs/flask_app_$last_folder.log" 2>&1 &

echo "Waiting for Flask app to initialize..."
while ! curl -s http://localhost:5000/health; do
  echo "Waiting for Flask app to be ready..."
  sleep 5
done

command="java -jar chatunitest-standalone.jar $1 project"

execute_command "$command" $1 $last_folder


echo "All files processed. The system will exit now."
