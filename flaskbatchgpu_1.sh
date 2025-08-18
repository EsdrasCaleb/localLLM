#!/bin/bash
#SBATCH --job-name=flask_uni        # Job name
#SBATCH --output=flask_uni_%j.log    # Log file (%j = job ID)
#SBATCH --time=2-00:00:00            # Test greater model in 2 days
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --mail-user=esdras.caleb@ufrn.br
#SBATCH --mail-type=ALL
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

#rm -r /tmp/chatunitest-info/hft-bomberman_inttrue
# Function to execute a command and capture its output
execute_command() {
  local command="$1"
  local env_file="$2"
  local file="$3"

  # sanitiza o nome do arquivo (só letras, números e _)
  local safe_file
  safe_file=$(echo "$file" | tr -cd '[:alnum:]_')

  # garante que a pasta exista
  mkdir -p unilogs

  # debug
  echo "DEBUG file=[$file] safe_file=[$safe_file] env_file=[$env_file]" >> debugunilogs.log

  local start_time=$(date +%s)
  echo "Executing: $command"
  local output
  output=$(eval "$command" 2>&1)
  local exit_code=$?

  if [ $exit_code -eq 0 ]; then
    echo "Successful execution of $env_file" >> "unilogs/executions_${safe_file}.log"
    echo -e "\nLog of $env_file:\n$output\n" >> "unilogs/logs_${safe_file}.log"
    #rm "$env_file"
  else
    echo "Problem in execution of $env_file" >> "unilogs/executions_${safe_file}.log"
    echo -e "\nError log of $env_file:\n$output\n" >> "unilogs/logs_${safe_file}.log"
  fi

  local end_time=$(date +%s)
  local elapsed_time=$((end_time - start_time))
  echo "Processing $env_file took $elapsed_time seconds" >> timings.log
}

# Run main.py in the background
python main.py >> "flask_app_gpu_$(date '+%Y-%m-%d_%H_%M_%S').log" 2>&1 &

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
        if execute_command "$command" "$env_file" "$last_folder"; then
          # Só executa isso se o comando acima tiver sucesso
          target_dir="executed/$(basename "$folder")"
          mkdir -p "$target_dir"

          mv "$env_file" "$target_dir/"
          echo "Moved file $env_file to $target_dir/"
          echo "Readed file $env_file"
        else
          echo "Erro ao processar $env_file. Não movido."
        fi
      fi
    done
    echo "Clear Models"
    python clear_models.py
  else
    echo "Directory $folder does not exist. Skipping..."
  fi
done

#rm -r ../scratch/models
echo "All files processed. The system will exit now."
