import os
import re

def create_env_files(projects_dir, models_file):
  """
  Creates .env files for each project in the given directory.

  Args:
    projects_dir: Path to the directory containing project folders.
    models_file: Path to the file containing model and URL pairs.
  """

  # Load model and URL pairs from models_file
  model_urls = {}
  with open(models_file, "r") as f:
    for line in f:
      parts = line.strip().split()
      if len(parts) == 2:
        model, url = parts
        model_urls[model] = url
      elif len(parts) == 1:
        model_urls[parts[0]] = "http://localhost:5000/generate_model"

  # Get project folders
  projects = [f for f in os.listdir(projects_dir) 
              if os.path.isdir(os.path.join(projects_dir, f)) 
              and f not in ["lib", "classes.txt"]]
  for project in projects:
      index = project.split("_")[0]
      project_name = project.split("_")[1]
      project_dir = os.path.join("./enfiles", project)
      os.makedirs(project_dir, exist_ok=True)
      for model,url in model_urls.items():
        env_file_path = os.path.join(project_dir, f"{model.replace('/','_')}_env")

        with open(env_file_path, "w") as f:
          with open("template.env", "r") as template:
              for line in template:
                line = line.replace("{index}", index)
                line = line.replace("{project}", project_name)

                line = line.replace("{url}", url)
                line = line.replace("{model}", model)

              f.write(line)

# Example usage
projects_dir = "../SF110"
models_file = "models.txt"
create_env_files(projects_dir, models_file)