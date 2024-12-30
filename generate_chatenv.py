import os
import re
from auxfunctions import load_env_file,download_model,file_repo
file_path = '.env'
env_data = load_env_file(file_path)


def load_project_path(project, projects_dir,classfile):
    """
    Loads project path from classes.txt file for a specific project.
    
    Args:
        project: Project name (e.g. "2_a4j")
        projects_dir: Path to the directory containing classes.txt
        
    Returns:
        str: Common package path prefix for the project
    """
    classes_file = os.path.join(projects_dir, "classes.txt")

    project_classes = []
    for line in classfile.split('\n'):
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and parts[0] == project:
            project_classes.append(parts[1])
           
    common_prefix = os.path.commonprefix(project_classes)
    if not project_classes:
        return ""
        
    # Find the common prefix of all class paths
    common_prefix = os.path.commonprefix(project_classes)
    # Remove the last partial package/class name if present
    if not common_prefix.endswith('.'):
        common_prefix = common_prefix.rsplit('.', 1)[0] + '.'
    if common_prefix.endswith('.'):  
      common_prefix = common_prefix[:-1]
    return common_prefix

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
    projects = {}
    class_file=""
    with open(os.path.join(projects_dir, "classes.txt"), 'r') as data:
      class_file = data.read()
    for  project in os.listdir(projects_dir) :
      if os.path.isdir(os.path.join(projects_dir, project)) and project not in ["lib", "classes.txt"]:
        # Create the target folder inside the project folder
        target_folder_path = os.path.join(projects_dir,project, "target")

        # Create the folder if it does not exist
        if not os.path.exists(target_folder_path):
            os.makedirs(target_folder_path)
        projects[project] = load_project_path(project,projects_dir,class_file)
    indexname = 0
    for model, url in model_urls.items():
        if model.endswith(".gguf"):
            file_name = model
            download_model(model_name=file_repo[file_name], file=file_name)
        else:
            download_model(model)
        model_ar = model.split("/")
        model_name = model_ar[-1]
        model_dir = os.path.join("./enfiles", f"{indexname:03}_{model_name}")
        indexname += 1
        #model_dir = os.path.join("./enfiles", model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        for project,project_path in projects.items():
            index = project.split("_")[0]
            project_name = project.split("_")[1]
            
            for intention in ["true", "false"]:
                env_file_path = os.path.join(model_dir, f"{project}_int{intention}_env")
                with open(env_file_path, "w") as f:
                    with open("template.env", "r") as template:
                        for line in template:
                            line = line.replace("{index}", index)
                            line = line.replace("{project}", project_name)
                            line = line.replace("{intention}", intention)
                            line = line.replace("{url}", url)
                            line = line.replace("{model}", model)
                            line = line.replace("{model_name}", model_name)
                            line = line.replace("{project_path}", project_path)
                            if url == "https://api.openai.com/v1/chat/completions":
                              line = line.replace("XXXKEYXXX", env_data.get('gpt_key',"XXXKEYXXX"))
                            f.write(line)

# Example usage
projects_dir = "../SF110"
models_file = "models.txt"
create_env_files(projects_dir, models_file)