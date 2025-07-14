import csv
import json
import os
import sys
import subprocess
import glob
import pandas as pd
import re
import lizard
import javalang
import time
import requests
import psutil
import threading

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
timeout={timeout}
baseDir=../SF110/{index}_{project}
groupId={project_path}
chatunitest-tests=../SF110/{index}_{project}/chatunitest-tests_{model_name}/
benchmark_file=data/{project}_{model_name}.csv
artifactId={project}
compileSourceRoots=../SF110/{index}_{project}/src/main/java
buildPath=../SF110/{index}_{project}/target
artifactPath=../SF110/{index}_{project}/{project}.jar
classPaths=../SF110/lib/evosuite.jar:../SF110/lib/:/tmp/chatunitest-info/{project}/build/{project_path_dir}/{project}/data/:../SF110/{index}_{project}:../SF110/{index}_{project}/lib:../SF110/{index}_{project}/test-lib:../SF110/{index}_{project}/target:./src/main/resources/dependency
packaging=jar
"""

def load_or_create_env(env_path=".env"):
  if not os.path.exists(env_path):
    with open(env_path, 'w') as f:
      f.write("# .env file created\n")

  env_dict = {}
  with open(env_path, 'r') as f:
    for line in f:
      # Remove leading/trailing whitespace and newline characters
      line = line.strip()

      # Ignore comments and empty lines
      if line and not line.startswith("#"):
        key_value = line.split("=", 1)

        # Ensure there are exactly two parts: key and value
        if len(key_value) == 2:
          key, value = key_value
          key = key.strip()
          value = value.strip()

          # Optionally, interpret booleans and numbers
          if value.lower() in ["true", "false"]:
            value = value.lower() == "true"
          elif value.isdigit():
            value = int(value)

          env_dict[key] = value

    return env_dict

env_dict = load_or_create_env()

def add_env_variable(key, value, env_path=".env"):
  env_dict[key] = value
  if not os.path.exists(env_path):
    load_or_create_env(env_path)

  lines = []
  found = False

  with open(env_path, 'r') as f:
    lines = f.readlines()

  for i, line in enumerate(lines):
    if line.strip().startswith(f"{key}="):
      lines[i] = f"{key}={value}\n"
      found = True
      break

  if not found:
    lines.append(f"{key}={value}\n")

  with open(env_path, 'w') as f:
    f.writelines(lines)



def run_chattester(env_path, command, justtest=False):
    global flask_process
    runner_env = load_or_create_env(env_path)
    # Início da medição
    start_time = time.time()
    usage_data = []

    # Captura o processo do Flask (se existir)
    flask_ps = psutil.Process(flask_process.pid) if flask_process else None

    # Lança o processo Java e monitora durante execução
    with subprocess.Popen(["java", "-jar", "chatunitest-standalone.jar", env_path, *command]) as proc:
        java_ps = psutil.Process(proc.pid)
        core_count = psutil.cpu_count()

        try:
            while proc.poll() is None:
                record = {
                    "timestamp": time.time(),
                    "java_cpu": java_ps.cpu_percent(interval=0.1)/core_count,
                    "java_mem": java_ps.memory_info().rss / (1024 * 1024),
                }

                if flask_ps:
                    try:
                        record["flask_cpu"] = flask_ps.cpu_percent(interval=0.1)/core_count
                        record["flask_mem"] = flask_ps.memory_info().rss / (1024 * 1024)
                    except psutil.NoSuchProcess:
                        record["flask_cpu"] = record["flask_mem"] = 0
                else:
                    record["flask_cpu"] = record["flask_mem"] = 0

                usage_data.append(record)
                time.sleep(0.1)  # ajuste esse intervalo conforme necessário
        except Exception as e:
            proc.kill()
            raise e

    # Tempo total
    end_time = time.time()
    total_time = end_time - start_time



    iteraction = 2
    if (not os.path.exists(runner_env["benchmark_file"])):
        print("Error benchmark not created")
        return -1
    else:
        # Read the last line and get the third column as integer
        with open(runner_env["benchmark_file"], newline='') as csvfile:
            reader = list(csv.reader(csvfile))
            if reader:
                last_line = reader[-1]
                if len(last_line) >= 3:
                    iteraction = int(last_line[4])
                else:
                    raise ValueError("Last line has fewer than 3 columns")
            else:
                raise ValueError("CSV file is empty")

    # Salvar métricas de uso
    usage_file = "usages_new.csv"
    write_header = (not os.path.exists(usage_file))
    with open(usage_file, "a") as f:
        if(write_header):
            f.write("model,timestamp(seconds),java_cpu(%),java_mem(MB),model_cpu(%),model_mem(MB),iteractions\n")
        max_model_cpu =0
        max_model_ram = 0
        max_java_cpu = 0
        max_java_ram = 0
        for row in usage_data:
            if(max_java_cpu < row["java_cpu"]):
                max_java_cpu = row["java_cpu"]
            if(max_java_ram < row["java_mem"]):
                max_java_ram = row["java_mem"]
            if(max_model_cpu < row["flask_cpu"]):
                max_model_cpu = row["flask_cpu"]
            if(max_model_ram < row["flask_mem"]):
                max_model_ram = row["flask_mem"]
            timestamp = row["timestamp"] - start_time
            #f.write(f"{runner_env['model']},{timestamp:.2f},{row['java_cpu']:.2f},{row['java_mem']:.2f},"
            #        f"{row['flask_cpu']:.2f},{row['flask_mem']:.2f},-\n")
        f.write(
            f"{runner_env['model']},{total_time:.2f},{max_java_cpu:.2f},{max_java_ram:.2f},"
            f"{max_model_cpu:.2f},{max_model_ram:.2f},{iteraction} \n")
    if justtest:
        return

    # Processamento pós-benchmark
    p_dt, semll_dt = generate_dt_smell(runner_env["benchmark_file"])
    smell_file = os.path.splitext(os.path.basename(runner_env["benchmark_file"]))[0]+"smell.csv"
    semll_dt.to_csv(smell_file, index=False, header=False)
    ts_df = run_test_smell_detector(smell_file)
    os.remove(smell_file)
    finaldt = merge_test_data(p_dt, ts_df)
    finaldt[['lizard_nloc', 'lizard_ccn', 'lizard_token', 'lizard_function_count']] = finaldt['file'].apply(
        lambda x: pd.Series(analyze_code_metrics(x)))
    finaldt[['total_assertion', 'methods_without_assertions', 'total_methods']] = finaldt['file'].apply(
        lambda x: pd.Series(count_assertions_in_methods(x)))
    final_file = "benchmarkfiles/"+os.path.splitext(os.path.basename(runner_env["benchmark_file"]))[0]+"smell.csv"
    if not os.path.exists("benchmarkfiles"):
        os.makedirs("benchmarkfiles")
    finaldt.to_csv(final_file, index=False)

def run_chattester_old(env_path, command, justtest=False):
  subprocess.run(["java", "-jar", "chatunitest-standalone.jar",env_path,*command], check=True)
  if(justtest):
      return
  runner_env = load_or_create_env(env_path)
  if (not os.path.exists(runner_env["benchmark_file"])):
      print("Error benchmark not created")
      return -1
  p_dt, semll_dt = generate_dt_smell(runner_env["benchmark_file"])
  smell_file = os.path.splitext(os.path.basename(runner_env["benchmark_file"]))[0]+"smell.csv"
  semll_dt.to_csv(smell_file, index=False, header=False)
  ts_df = run_test_smell_detector(smell_file)
  os.remove(smell_file)
  finaldt = merge_test_data(p_dt, ts_df)
  finaldt[['lizard_nloc', 'lizard_ccn', 'lizard_token', 'lizard_function_count']] = finaldt['file'].apply(
      lambda x: pd.Series(analyze_code_metrics(x)))
  finaldt[['total_assertion', 'methods_without_assertions', 'total_methods']] = finaldt['file'].apply(
      lambda x: pd.Series(count_assertions_in_methods(x)))
  final_file = "benchmarkfiles/"+os.path.splitext(os.path.basename(runner_env["benchmark_file"]))[0]+"smell.csv"
  if not os.path.exists("benchmarkfiles"):
      os.makedirs("benchmarkfiles")
  finaldt.to_csv(final_file, index=False)

def generate_dt_smell(csv_path):
    df = pd.read_csv(csv_path)

    # Filtra só SUCCESS e arquivos que existem
    df = df[df['result'] == 'SUCCESS']
    df = df[df['file'].apply(lambda x: os.path.exists(x))]

    rows = []
    second_rows = []

    for _, row in df.iterrows():
        project = row['project']
        model = row['model']
        file_path = row['file']
        test_number = row['test_number']
        num_interactions = row['num_interactions']
        num_corrections = row['num_corrections']
        result = row['result']
        mutation_null = row['mutation_null']
        mutation_var = row['mutation_var']
        mutation_bool = row['mutation_bool']
        mutation_aritime = row['mutation_aritime']
        mutation_logic = row['mutation_logic']
        mutation_relat = row['mutation_relat']

        # Número de métodos SUT é fixo: 1
        number_of_sut_methods = 1

        # Conta número de @Test
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        number_of_tests = len(re.findall(r'@Test\b', content))

        rows.append({
            'project': project,
            'file': file_path,
            'num_interactions': num_interactions,
            'num_corrections': num_corrections,
            'result': result,
            'model': model,
            'test_number': test_number,
            'mutation_null': mutation_null,
            'mutation_var': mutation_var,
            'mutation_bool': mutation_bool,
            'mutation_aritime': mutation_aritime,
            'mutation_logic': mutation_logic,
            'mutation_relat': mutation_relat,
            'number_of_sut_methods': number_of_sut_methods,
            'number_of_tests': number_of_tests
        })

        # Gera caminho sut_path
        # file tem o caminho completo, pega a parte após chatunitest-tests_MODELNAME/
        split_token = f'chatunitest-tests_{model.replace("/", "_")}/'
        if split_token in file_path:
            path_arr = os.path.dirname(file_path).split(split_token)
            class_path_arr = row['class'].split('.')
            sut_class_name = class_path_arr[-1] + '.java'
            sut_path = os.path.join(path_arr[0], 'src', 'main', 'java',
                                    "/".join(class_path_arr[0:-1]), sut_class_name)
            if (not os.path.exists(sut_path)):
                print("Erro class " + sut_class_name + " not found in " + sut_path)
            second_rows.append([project, file_path, sut_path])
    df1 = pd.DataFrame(rows)
    df2 = pd.DataFrame(second_rows)

    return df1, df2


def run_test_smell_detector(input_csv_path,  jar_name="TestSmellDetector.jar"):
  subprocess.run(["java", "-jar", jar_name, input_csv_path], check=True)


  output_files = glob.glob(os.path.join("Output_TestSmellDetection_*.csv"))
  if not output_files:
    raise FileNotFoundError("Nenhum arquivo CSV foi gerado pelo TestSmellDetector.")

  output_csv = output_files[0]

  # Lê o CSV em um DataFrame
  df = pd.read_csv(output_csv)

  # Apaga o arquivo CSV gerado
  os.remove(output_csv)

  return df

def count_unique_methods_tested(evosuite_file, source_file):
  if not os.path.exists(evosuite_file) or not os.path.exists(source_file):
    return 0

  try:
    with open(source_file, 'r', encoding='utf-8') as f:
      sut_code = f.read()
  except UnicodeDecodeError:
    with open(source_file, 'r', encoding='latin1') as f:
      sut_code = f.read()

  sut_tree = javalang.parse.parse(sut_code)

  sut_methods = set()
  for _, node in sut_tree.filter(javalang.tree.MethodDeclaration):
    sut_methods.add(node.name)

  try:
    with open(evosuite_file, 'r', encoding='utf-8') as f:
      test_code = f.read()
  except UnicodeDecodeError:
    with open(evosuite_file, 'r', encoding='latin1') as f:
      test_code = f.read()

  methods_tested = {m for m in sut_methods if f".{m}(" in test_code}
  return len(methods_tested)

def find_existing_evosuite_tests(projects_dir):
  import glob

  result = {}
  data = []

  for project in os.listdir(projects_dir):
    if not (project.startswith(tuple(f"{i}_" for i in range(1, 8)))):
      continue

    project_path = os.path.join(projects_dir, project, "evosuite-tests")
    if not os.path.isdir(project_path):
      continue

    project_name = project.split("_", 1)[1]
    existing_files = set()

    for evo_path in glob.glob(os.path.join(project_path, "**", "*EvoSuiteTest.java"), recursive=True):
      if evo_path not in existing_files:
        existing_files.add(evo_path)

        class_rel_path = os.path.relpath(evo_path, project_path).replace("EvoSuiteTest.java", ".java")
        sut_path = os.path.join(projects_dir, project, "src", "main", "java", class_rel_path)

        if os.path.isfile(sut_path):
          data.append([project_name, evo_path, sut_path])
        else:
          print(f"error cant find destination class to {evo_path}")

    if existing_files:
      result[project_name] = list(existing_files)

  return result, data


def find_existing_evosuite_tests_chat(projects_dir):
    result = {}
    data = []
    with open('chattstermapping1_7.json', 'r') as f:
        projects_chattester_mapping = json.load(f)

    for project, classes in projects_chattester_mapping.items():
        project_path = os.path.join(projects_dir, project, "evosuite-tests")
        existing_files = []
        project_name = project.split("_")[1]

        for class_path in classes:
            relative_path = os.path.join(*class_path.split('.')) + "EvoSuiteTest.java"
            evo_path = os.path.join(project_path, relative_path)

            if os.path.isfile(evo_path):
                sut_relative = os.path.join("src", "main", "java", *class_path.split('.')) + ".java"
                sut_path = os.path.join(projects_dir, project, sut_relative)
                if(os.path.isfile(sut_path)):
                  if(not evo_path in existing_files):
                    existing_files.append(evo_path)
                    data.append([project_name, evo_path, sut_path])
                else:
                  print("error cant find destination class to "+evo_path)

        if existing_files:
            result[project_name] = existing_files

    return result,data


def merge_test_data(final_dt, smell_dt):
  # Faz o merge usando 'file' de df1 e 'TestFilePath' de df2
  merged_df = final_dt.merge(
    smell_dt,
    left_on="file",
    right_on="TestFilePath",
    how="inner"
  )

  # Seleciona e reordena as colunas conforme especificado
  final_columns = [
    "project", "file", "num_interactions", "num_corrections", "result", "model", "test_number",
    "mutation_null", "mutation_var", "mutation_bool", "mutation_aritime", "mutation_logic", "mutation_relat",
    "number_of_sut_methods","number_of_tests","NumberOfMethods", "Assertion Roulette", "Conditional Test Logic",
    "Constructor Initialization","Default Test", "EmptyTest", "Exception Catching Throwing", "General Fixture",
    "Mystery Guest", "Print Statement","Redundant Assertion", "Sensitive Equality", "Verbose Test", "Sleepy Test",
    "Eager Test", "Lazy Test","Duplicate Assert", "Unknown Test", "IgnoredTest", "Resource Optimism",
    "Magic Number Test", "Dependent Test"
  ]

  return merged_df[final_columns]


def generate_dt_from_evosuite_files(existing_files_by_project):
  data = []

  # Montar todos os caminhos sut_paths_by_project
  sut_paths_by_project = {}
  for project in existing_files_by_project.keys():
    filename = f"evosuittestsemll_{project}"
    if os.path.exists(filename):
      df_sut = pd.read_csv(filename, header=None, names=["project", "file", "source_file"])
      for _, row in df_sut.iterrows():
        sut_paths_by_project[row["file"]] = row["source_file"]
    else:
      print(f"Arquivo {filename} não encontrado para o projeto {project}")

  for project, files in existing_files_by_project.items():
    for file_path in files:
      try:
        with open(file_path, 'r', encoding='utf-8') as f:
          content = f.read()
        num_tests = len(re.findall(r'@Test\b', content))
      except Exception as e:
        print(f"Erro ao ler {file_path}: {e}")
        num_tests = -1

      source_file = sut_paths_by_project.get(file_path, None)
      if source_file:
        num_sut_methods = count_unique_methods_tested(file_path, source_file)
      else:
        num_sut_methods = 0

      data.append([
        project,  # project
        file_path,  # file
        1,  # num_interactions
        0,  # num_corrections
        "SUCCESS",  # result
        "evosuite",  # model
        0,  # test_number
        -1, -1, -1, -1, -1, -1,  # mutation_* colunas
        num_sut_methods,
        num_tests
      ])

  df = pd.DataFrame(data, columns=[
    "project", "file", "num_interactions", "num_corrections", "result", "model",
    "test_number", "mutation_null", "mutation_var", "mutation_bool",
    "mutation_aritime", "mutation_logic", "mutation_relat",
    "number_of_sut_methods", "number_of_tests"
  ])

  return df

def analyze_code_metrics(file_path):
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

  analysis = lizard.analyze_file(file_path)
  return analysis.nloc, analysis.CCN, analysis.token_count, len(analysis.function_list)


def count_assertions_in_methods(java_file):
  if not os.path.exists(java_file):
    raise FileNotFoundError(f"Arquivo não encontrado: {java_file}")

  with open(java_file, 'r', encoding='utf-8') as file:
    code = file.read()
  tree = javalang.parse.parse(code)

  total_assertions = 0
  methods_without_assertions = 0
  total_methods = 0

  for _, node in tree.filter(javalang.tree.MethodDeclaration):
    total_methods += 1
    assertion_count = sum(1 for _, stmt in node.filter(javalang.tree.MethodInvocation) if
                          (stmt.member.startswith("assert") or stmt.member.startswith("fail")))

    total_assertions += assertion_count
    if assertion_count == 0:
      methods_without_assertions += 1

  return total_assertions, methods_without_assertions, total_methods

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
    global PROJECTS_DIR
    while not os.path.exists(PROJECTS_DIR):
        print(f"\nBenchmark not found at {PROJECTS_DIR}.")
        print(f"Please download it from {BENCHMARK_URL}")
        print(f"Extract it to the root folder and rename it to 'SF110'.\n")

        # Ask the user for the correct path until it exists
        PROJECTS_DIR = input("Please provide the correct path to the SF110 directory: ").strip()

        if os.path.exists(PROJECTS_DIR):
            print(f"Path exists: {PROJECTS_DIR}")
        else:
            print(f"Directory {PROJECTS_DIR} not found. Please try again.")
    print(f"Benchmark found at {PROJECTS_DIR}. Proceeding...")

def select_option():
    print("\nChoose an option:")
    print("a - Run a model evaluation")
    print("b - Run a project benchmark")
    print("c - Generate EvoSuite benchmark data")
    print("d - make the full benchmark")
    print("e - make the local tests")

    choice = input("Enter your choice (a/b/c): ").strip().lower()
    if choice not in ("a", "b", "c","d","e"):
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

    #colocar so primeira parte
    print("\nAvailable models:")
    for i, model in enumerate(models, start=1):
        model_arr = model.split(" ")
        print(f"{i}. {model_arr[0]}")

    model_index = int(input("Select a model number: ")) - 1
    if model_index < 0 or model_index >= len(models):
        sys.exit("Invalid model choice. Exiting.")

    selected_model = models[model_index]

    api_key = "XXXXXXX"
    if model_type == "web":
      if((model_index==0 or model_index==4)):
          if("g_tokens" not in env_dict):
            print("You don't have Google Gemini API keys configured.")
            print("Generate your Gemini API keys here:")
            print("  - https://aistudio.google.com/app/apikey")
            keys = input("Paste your Gemini API keys here, separated by commas if you have more than one: ").strip()
            add_env_variable("g_tokens", keys)
          api_key = env_dict["g_tokens"]
      if (model_index == 1 or model_index == 3):
        if("MISTRAL_API_KEY" not in env_dict):
          print("You don't have a Mistral API key configured.")
          print("Generate your Mistral API key here:")
          print("  - https://console.mistral.ai/")
          key = input("Paste your Mistral API key here, separated by commas if you have more than one: ").strip()
          add_env_variable("MISTRAL_API_KEY", key)
        api_key = env_dict["MISTRAL_API_KEY"]
      if (model_index == 2):
        if "gpt_key" not in env_dict:
          print("You don't have an OpenAI GPT API key configured.")
          print("Generate your OpenAI API key here:")
          print("  - https://platform.openai.com/account/api-keys")
          key = input("Paste your OpenAI API key here, separated by commas if you have more than one: ").strip()
          add_env_variable("gpt_key", key)
        api_key = env_dict["gpt_key"]
      if (model_index == 5):
        if "CHUTES_API_KEY" not in env_dict:
          print("You don't have a Chutes.ai API key configured.")
          print("Generate your Chutes.ai API key here:")
          print("  - https://chutes.ai/app/api")
          key = input("Paste your Chutes.ai (the fee varies) API key here, separated by commas if you have more than one: ").strip()
          add_env_variable("CHUTES_API_KEY", key)
          env_dict["CHUTES_API_KEY"] = key
        api_key = env_dict["CHUTES_API_KEY"]
    else:
      if "HF_TOKEN" not in env_dict:
        print("You don't have a HuggingFace access token configured.")
        print("Generate one here: https://huggingface.co/settings/tokens")
        token = input("Paste your HuggingFace token here: ").strip()
        add_env_variable("HF_TOKEN", token)

    return selected_model, api_key


def get_projects():
    try:
        all_dirs = [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]

        # Filter projects that start with a number followed by underscore
        numbered_projects = []
        for project in all_dirs:
            parts = project.split('_', 1)
            if len(parts) == 2:
                try:
                    number = int(parts[0])
                    numbered_projects.append((number, project))
                except ValueError:
                    continue

        # Sort projects by number
        numbered_projects.sort()

        if not numbered_projects:
            sys.exit(f"No numbered projects found in {PROJECTS_DIR}. Exiting.")
    except FileNotFoundError:
        sys.exit(f"Error: {PROJECTS_DIR} not found.")

    # Filter to show only projects 1-7
    projects_1_to_7 = []
    print("\nAvailable projects:")
    for number, project in numbered_projects:
        if number <= 7:
            projects_1_to_7.append(project)
            print(f"{number}. {project}")

    if not projects_1_to_7:
        sys.exit("No projects with numbers 1-7 found. Exiting.")

    selected_project = input("Enter project number: ").strip()

    try:
        selected_number = int(selected_project)
    except ValueError:
        sys.exit("Invalid input. Must be a number.")

    for num, proj in numbered_projects:
        if num == selected_number:
            return proj

    sys.exit("Selected project not found. Exiting.")


def generate_chatenv_file(project, model_string, api_key):
    model_arr = model_string.split(" ")
    time_out = 30
    project_arr = project.split("_")
    model_name = model_arr[0].replace("/", "_")
    with open('chattstermapping1_7.json', 'r') as f:
        projects_chattester_mapping = json.load(f)
    input_string =next(iter(projects_chattester_mapping[project]))
    last_dot_index = input_string.rfind('.')
    up_to_last_dot = input_string[:last_dot_index + 1]

    # Convert dots to slashes
    converted_to_slashes = input_string.replace('.', '/')
    if(len(model_arr)>2):
        time_out = model_arr[2]
    env_content = ENV_TEMPLATE.format(
        api_key=api_key,
        url=model_arr[1],
        model=model_arr[0],
        timeout=time_out,
        intention="true",
        temp=0.7,
        project=project_arr[1],
        index=project_arr[0],
        project_path=up_to_last_dot,
        project_path_dir=converted_to_slashes,
        model_name=model_name
    )
    # Check if the folder exists, if not, create it
    if not os.path.exists("enfiles"):
        os.makedirs("enfiles")
    env_filename = f"enfiles/{project}_{model_name}.env"
    with open(env_filename, "w") as file:
        file.write(env_content)
    return env_filename
    print(f"Generated {env_filename}")

env_data =load_or_create_env(env_path=".env")

def get_model_projects(select_project=False):
  model, api_key = get_models()
  project = "1_tullibee"
  if(select_project):
    project = get_projects()
  envfile = generate_chatenv_file(project, model, api_key)
  return envfile

flask_process = None

def stream_output(pipe, prefix):
    for line in iter(pipe.readline, b''):
        print(f"[{prefix}] {line.decode().rstrip()}")
    pipe.close()

# Function to start Flask server in background
def start_flask_server():
    global flask_process
    flask_process = subprocess.Popen(
        ['python', 'main.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1
    )

    # Iniciar threads para imprimir stdout e stderr
    threading.Thread(target=stream_output, args=(flask_process.stdout, 'STDOUT'), daemon=True).start()
    threading.Thread(target=stream_output, args=(flask_process.stderr, 'STDERR'), daemon=True).start()

    # Esperar o servidor subir
    status = False
    while not status:
        try:
            response = requests.get("http://localhost:5000/health")
            if response.status_code == 200:
                print("Server is running.")
                status = True
                break
            else:
                print(f"Error: Server returned status code {response.status_code}.")
        except requests.exceptions.RequestException:
            print("Waiting server to start...")
        time.sleep(3)
    return flask_process

def execute_model(array_command):
    enfile = get_model_projects(len(array_command) == 1)
    return execute_benchmark(enfile, array_command,True)

def generate_model_benchmark(model,api_key):
    all_dirs = [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]

    for project in all_dirs:
        parts = project.split('_', 1)
        if len(parts) == 2:
            try:
                number = int(parts[0])
            except ValueError:
                continue
            if number <= 7:
                execute_benchmark(generate_chatenv_file(project, model, api_key),["project"])

def execute_benchmark(enfile, array_command,justtest=False):
    global flask_process
    runner_env = load_or_create_env(enfile)
    if (runner_env["url"].startswith('http://localhost:5000')):
        response = None
        if(flask_process):
            try:
                response = requests.get("http://localhost:5000/clear_models")
            except Exception as e:
                print(e)
        if response and response.status_code == 200:
            print("Server is running.")
        else:
            print("Start flask")
            print(flask_process)
            start_flask_server()
    return run_chattester(enfile, array_command,justtest)

def main():
    check_requirements()
    check_benchmark()
    if len(sys.argv) > 1:
      option = sys.argv[1]
    else:
      option = select_option()
    # Check if the folder exists, if not, create it
    if not os.path.exists("data"):
        os.makedirs("data")
    array_command = []
    match(option):
      case "a":
        execute_model(['method','OrderState','equals'])
      case "b":
        execute_model(['project'])
      case "c":
        evosuite_data, smell_evo_data = find_existing_evosuite_tests_chat(PROJECTS_DIR)
        df = pd.DataFrame(smell_evo_data)
        #pd.set_option('display.max_colwidth', None)  # mostra conteúdo completo das colunas
        df.to_csv("evotssmell.csv", index=False, header=False)
        ts_df = run_test_smell_detector("evotssmell.csv")
        finaldt = generate_dt_from_evosuite_files(evosuite_data)
        finaldt = merge_test_data(finaldt,ts_df)
        finaldt[['lizard_nloc', 'lizard_ccn', 'lizard_token', 'lizard_function_count']] = finaldt['file'].apply(
          lambda x: pd.Series(analyze_code_metrics(x)))
        finaldt[['total_assertion', 'methods_without_assertions', 'total_methods']] = finaldt['file'].apply(
          lambda x: pd.Series(count_assertions_in_methods(x)))
        if not os.path.exists("benchmarkfiles"):
            os.makedirs("benchmarkfiles")
        finaldt.to_csv("benchmarkfiles/evosuite_final.csv", index=False)
        print("Evosuite files in 'data/evosuite_final.csv'")
      case "d":
            if "g_tokens" not in env_dict:
              print("You don't have Google Gemini API keys configured.")
              print("Generate your Gemini API keys here:")
              print("  - https://aistudio.google.com/app/apikey")
              keys = input("Paste your Gemini API keys here, separated by commas if you have more than one: ").strip()
              add_env_variable("g_tokens", keys)
            if ("MISTRAL_API_KEY" not in env_dict):
              print("You don't have a Mistral API key configured.")
              print("Generate your Mistral API key here:")
              print("  - https://console.mistral.ai/")
              key = input(
                  "Paste your Mistral API key here, separated by commas if you have more than one: ").strip()
              add_env_variable("MISTRAL_API_KEY", key)
            if "gpt_key" not in env_dict:
              print("You don't have an OpenAI GPT API key configured.")
              print("Generate your OpenAI API key here:")
              print("  - https://platform.openai.com/account/api-keys")
              key = input(
                  "Paste your OpenAI API key here, separated by commas if you have more than one: ").strip()
              add_env_variable("gpt_key", key)
            if "CHUTES_API_KEY" not in env_dict:
              print("You don't have a Chutes.ai API key configured.")
              print("Generate your Chutes.ai API key here:")
              print("  - https://chutes.ai/app/api")
              key = input(
                  "Paste your Chutes.ai (the fee varies) API key here, separated by commas if you have more than one: ").strip()
              add_env_variable("CHUTES_API_KEY", key)
            if "HF_TOKEN" not in env_dict:
              print("You don't have a HuggingFace access token configured.")
              print("Generate one here: https://huggingface.co/settings/tokens")
              token = input("Paste your HuggingFace token here: ").strip()
              add_env_variable("HF_TOKEN", token)
            print("Generating local models")
            WEB_MODELS_FILE
            try:
                with open(LOCAL_MODELS_FILE, "r") as file:
                    models = file.read().splitlines()
                    if not models:
                        sys.exit(f"No models found in {LOCAL_MODELS_FILE}. Exiting.")
            except FileNotFoundError:
                sys.exit(f"Error: {LOCAL_MODELS_FILE} not found.")

                # colocar so primeira parte
            for model in models:
                generate_model_benchmark(model,"XXXXX")
            print("Generating web models")
            models = []
            try:
                with open(WEB_MODELS_FILE, "r") as file:
                    models = file.read().splitlines()
                    if not models:
                        sys.exit(f"No models found in {WEB_MODELS_FILE}. Exiting.")
            except FileNotFoundError:
                sys.exit(f"Error: {WEB_MODELS_FILE} not found.")
            for model_index, model in enumerate(models, start=1):
                if ((model_index == 0 or model_index == 4)):
                    generate_model_benchmark(model, env_dict["g_tokens"])
                if (model_index == 1 or model_index == 3):
                    generate_model_benchmark(model, env_dict["MISTRAL_API_KEY"])
                if (model_index == 2):
                    generate_model_benchmark(model, env_dict["gpt_key"])
                if (model_index == 5):
                    generate_model_benchmark(model, env_dict["CHUTES_API_KEY"])
      case "e":
            print("Generating local models")
            try:
                with open(LOCAL_MODELS_FILE, "r") as file:
                    models = file.read().splitlines()
                    if not models:
                        sys.exit(f"No models found in {LOCAL_MODELS_FILE}. Exiting.")
            except FileNotFoundError:
                sys.exit(f"Error: {LOCAL_MODELS_FILE} not found.")

                # colocar so primeira parte
            for model in models:
                execute_benchmark(generate_chatenv_file("1_tullibee", model, "XXXXX"),
                                  ['method','OrderState','equals'],True)


    if flask_process:
        flask_process.terminate()  # Sends SIGTERM signal to the process
        flask_process.wait()  # Wait for the process to terminate
        print("Flask server stopped.")
    else:
        print("Flask server is not running.")

if __name__ == "__main__":
    main()
