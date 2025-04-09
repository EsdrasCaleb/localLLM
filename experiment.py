import os
import sys
import subprocess

REQUIREMENTS_FILE = "requirements.txt"
LOCAL_MODELS_FILE = "models_local.txt"
WEB_MODELS_FILE = "models_web.txt"
PROJECTS_DIR = "../SF110"
BENCHMARK_URL = "http://www.evosuite.org/files/SF110-20130704-src.zip"

ENV_TEMPLATE = """apiKeys={api_key}
url={url}
model={model}
enableMultithreading=false
parentEnvPath=
phase=BENCHMARK
sleeptime={timeout}
max_tokens=1024
use_intention={intention}
onlyUpdateClass=true
max_prompt_tokens=3000
temperature={temp}
plugin={project}
timeout=30
baseDir=../SF110/{index}_{project}
groupId={project_path}
chatunitest-tests=../SF110/{index}_{project}/chatunitest-tests_{model_name}/
benchmark_file=evosuitcsvs/evosuit_{project}.csv
artifactId={project}
compileSourceRoots=../SF110/{index}_{project}/src/main/java
buildPath=../SF110/{index}_{project}/target
artifactPath=../SF110/{index}_{project}/{project}.jar
classPaths=../SF110/lib/evosuite.jar:../SF110/lib/:/tmp/chatunitest-info/{project}/build/{project_path_dir}/{project}/data/:../SF110/{index}_{project}:../SF110/{index}_{project}/lib:../SF110/{index}_{project}/test-lib:../SF110/{index}_{project}/target:src/main/resources/dependency:target/classes
packaging=jar
noExecution=true
"""


def check_requirements():
    try:
        with open(REQUIREMENTS_FILE, "r") as file:
            packages = file.read().splitlines()

        missing_packages = []
        for package in packages:
            try:
                subprocess.check_output([sys.executable, "-m", "pip", "show", package.split("==")[0]])
            except subprocess.CalledProcessError:
                missing_packages.append(package)

        if missing_packages:
            print(f"Missing packages: {', '.join(missing_packages)}")
            install = input("Do you want to install them? (y/n): ").strip().lower()
            if install == "y":
                subprocess.run([sys.executable, "-m", "pip", "install", *missing_packages])
            else:
                sys.exit("Required packages are missing. Exiting.")

    except FileNotFoundError:
        sys.exit(f"Error: {REQUIREMENTS_FILE} not found.")


def check_benchmark():
    if not os.path.exists(PROJECTS_DIR):
        print(f"\nBenchmark not found at {PROJECTS_DIR}.")
        print(f"Please download it from {BENCHMARK_URL}")
        print(f"Extract it to the root folder and rename it to 'SF110'.\n")
        input("Press Enter after completing this step...")
        if not os.path.exists(PROJECTS_DIR):
            sys.exit("SF110 directory not found. Exiting.")


def select_option():
    print("\nChoose an option:")
    print("a - Run a single model benchmark")
    print("b - Run all benchmarks")
    print("c - Generate EvoSuite benchmark")

    choice = input("Enter your choice (a/b/c): ").strip().lower()
    if choice not in ("a", "b", "c"):
        sys.exit("Invalid choice. Exiting.")
    return choice


def get_models():
    model_type = input("Do you want to use a local or web model? (local/web): ").strip().lower()
    if model_type not in ("local", "web"):
        sys.exit("Invalid choice. Exiting.")

    models_file = LOCAL_MODELS_FILE if model_type == "local" else WEB_MODELS_FILE
    try:
        with open(models_file, "r") as file:
            models = file.read().splitlines()
            if not models:
                sys.exit(f"No models found in {models_file}. Exiting.")
    except FileNotFoundError:
        sys.exit(f"Error: {models_file} not found.")

    print("\nAvailable models:")
    for i, model in enumerate(models, start=1):
        print(f"{i}. {model}")

    model_index = int(input("Select a model number: ")) - 1
    if model_index < 0 or model_index >= len(models):
        sys.exit("Invalid model choice. Exiting.")

    selected_model = models[model_index]

    api_key = ""
    if model_type == "web":
        api_key = input("Enter the API key for the web model: ").strip()

    return selected_model, api_key


def get_projects():
    try:
        projects = [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]
        if not projects:
            sys.exit(f"No projects found in {PROJECTS_DIR}. Exiting.")
    except FileNotFoundError:
        sys.exit(f"Error: {PROJECTS_DIR} not found.")

    print("\nAvailable projects:")
    for i, project in enumerate(projects, start=1):
        print(f"{i}. {project}")

    selected_projects = input("Enter project numbers separated by commas: ").strip()
    selected_indices = [int(i) - 1 for i in selected_projects.split(",")]

    for i in selected_indices:
        if i < 0 or i >= len(projects):
            sys.exit("Invalid project selection. Exiting.")

    return [projects[i] for i in selected_indices]


def generate_env_file(project, model, api_key):
    env_content = ENV_TEMPLATE.format(
        api_key=api_key or "XXXKEYXXX",
        url="https://example.com",  # Modify as needed
        model=model,
        timeout=30,
        intention="true",
        temp=0.7,
        project=project,
        index=1,  # Modify as needed
        project_path=project,
        project_path_dir=project,
        model_name=model
    )

    env_filename = f"{project}.env"
    with open(env_filename, "w") as file:
        file.write(env_content)

    print(f"Generated {env_filename}")


def main():
    check_requirements()
    check_benchmark()
    option = select_option()
    model, api_key = get_models()
    projects = get_projects()

    for project in projects:
        generate_env_file(project, model, api_key)


if __name__ == "__main__":
    main()
