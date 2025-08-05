import os
import pandas as pd
import glob
import re
import subprocess
import gspread
import lizard
import javalang
from oauth2client.service_account import ServiceAccountCredentials

# Configuration
csv_folder = "./result"
output_csv = "test_files.csv"
jar_path = "./TestSmellDetector.jar"
spreadsheet_id = "i1rx60OB44n8nWCsziMVbqy9oa9z5wfliQ8QOGfSbFxMU"
worksheet_name = "results"

# 1. Load and filter CSV files
all_files = glob.glob(os.path.join(csv_folder, "benchmarkresults_*.csv"))


# Function to fix lines that contain ")" followed by a ","
def fix_csv_format(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    fixed_lines = []
    for line in lines:
        if re.search(r"\),", line): # Only process lines that match the pattern
            line = re.sub(r"\)(,)", r')"\1', line, count=1)
            line = re.sub(r"^(.*?,.*?),", r'\1,"', line, count=1)
        fixed_lines.append(line)

    # Overwrite the file with the fixed content
    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(fixed_lines)

# Fix all CSV files
for csv_file in all_files:
    fix_csv_format(csv_file)



# Function to extract root_path from file path
def extract_root_path(file_path):
    parts = file_path.split("/")
    return "/".join(parts[:3]) + "/" if len(parts) >= 3 else file_path

# 1. Load CSVs into a single DataFrame
df_list = [pd.read_csv(f, on_bad_lines='warn') for f in all_files]

df = pd.concat(df_list, ignore_index=True)
df_result = df[df['result'] == 'SUCCESS'].drop_duplicates(subset=['file'])
df_result['source_file'] = df_result.apply(
    lambda row: f"{row['file'].split('/chatunitest-tests_')[0]}/src/main/java/{row['class'].replace('.', '/')}.java",
    axis=1
)
# 2. Generate intermediate CSV
df_result[['project', 'file','source_file']].to_csv(output_csv, index=False, header=False)

# 3. Run TestSmellDetector.jar and wait for the process to finish
subprocess.run(["java", "-jar", jar_path, output_csv], check=True)

# 4. Load TestSmellDetector output (use the first file found)
output_files = glob.glob("Output_TestSmellDetection_*.csv")
if output_files:
    output_test_smell = output_files[0]  # Use the first output file
    smell_df = pd.read_csv(output_test_smell)
    columns_to_merge = [
        "NumberOfMethods", "Assertion Roulette", "Conditional Test Logic", "Constructor Initialization", "Default Test",
        "EmptyTest", "Exception Catching Throwing", "General Fixture", "Mystery Guest", "Print Statement", "Redundant Assertion",
        "Sensitive Equality", "Verbose Test", "Sleepy Test", "Eager Test", "Lazy Test", "Duplicate Assert", "Unknown Test",
        "IgnoredTest", "Resource Optimism", "Magic Number Test", "Dependent Test"
    ]
    df_result = df_result.merge(smell_df[['TestFilePath'] + columns_to_merge], left_on='file', right_on='TestFilePath', how='left')
    df_result.drop(columns=['TestFilePath'], inplace=True)

# 5. Static analysis with Lizard and assertion counting
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
        assertion_count = sum(1 for _, stmt in node.filter(javalang.tree.MethodInvocation) if (stmt.member.startswith("assert") or stmt.member.startswith("fail")) )

        total_assertions += assertion_count
        if assertion_count == 0:
            methods_without_assertions += 1

    return total_assertions, methods_without_assertions, total_methods

# Apply analysis functions
df_result[['lizard_nloc', 'lizard_ccn', 'lizard_token', 'lizard_function_count']] = df_result['file'].apply(lambda x: pd.Series(analyze_code_metrics(x)))
df_result[['total_assertion', 'methods_without_assertions', 'total_methods']] = df_result['file'].apply(lambda x: pd.Series(count_assertions_in_methods(x)))

# 6. Create final DataFrame excluding prompt, class and method (if they exist)
df_final = df_result.drop(columns=['prompt', 'class', 'method'], errors='ignore')

backup_csv = "backup_results2.csv"

# Save local backup before sending to Google Sheets
df_final.to_csv(backup_csv, index=False)
print(f"Backup salvo em {backup_csv}")

print(f"Os dados foram salvos localmente em {backup_csv}")

