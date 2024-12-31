#!/bin/bash


# Loop through each folder in the "envs" directory
for folder in enfiles/*; do
  if [ -d "$folder" ]; then
    echo "Processing folder: $folder"

    # Loop through each .env file in the folder
    for env_file in "$folder"/*; do
      # Construct the command
      # Get the first idle partition
      idle_partition=$(sinfo --format="%P %D %T" | awk '$3 == "idle" {print $1; exit}')

      # Check if an idle partition was found
      if [ -z "$idle_partition" ]; then
        echo "No idle partition available. Exiting."
        exit 1
      fi

      echo "Using idle partition: $idle_partition in file $env_file and folder $folder"
      # Execute the command and capture output
      sbatch --partition="$idle_partition" flaskbatchenv.sh $env_file $folder
      exit 0
      break
    done
    break
  fi
done

echo "All files processed in batch queue."
