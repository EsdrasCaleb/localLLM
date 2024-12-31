#!/bin/bash

MAX_JOBS=4  # Maximum number of jobs per partition
echo "Processing folder: 009_deepseek-coder-1.3b-instruct"
# Loop through each folder in the "envs" directory
for env_file in enfiles/009_deepseek-coder-1.3b-instruct/*; do
    while true; do
      # Get a list of all idle partitions excluding those starting with "gpu" or "fpga"
      idle_partitions=($(sinfo --format="%P %T" | awk '$2 == "idle" && $1 !~ /^(gpu|fpga)/ {print $1}'))

      # Check if there are any idle partitions
      if [ ${#idle_partitions[@]} -eq 0 ]; then
        echo "No idle partitions available. Retrying in 10 seconds."
        exit 1
      fi

      selected_partition=""

      # Check each idle partition for user job count
      for partition in "${idle_partitions[@]}"; do
        job_count=$(squeue --user="$USER" --partition="$partition" --noheader | wc -l)
        if [ "$job_count" -lt "$MAX_JOBS" ]; then
          selected_partition="$partition"
          break
        fi
      done

      if [ -z "$selected_partition" ]; then
        echo "All idle partitions are full of user jobs. Retrying in 3 hours."
        exit 0
      else
        echo "Using partition: $selected_partition with $job_count jobs running by user $USER."
        echo "Submitting file $env_file from folder $folder."
        sbatch --partition="$selected_partition" flaskbatchenv.sh "$env_file" "$folder"
        break
      fi
    done
done

echo "All files processed in batch queue."
