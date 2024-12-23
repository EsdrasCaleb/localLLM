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

# Activate virtual environment (if needed)
source vllm_env/bin/activate  # Update with your virtual environment path

# Run Flask with the model
python main.py --host=0.0.0.0 --port=8080
