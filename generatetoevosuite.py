import os
import pandas as pd
import lizard
import javalang

# Carregar o CSV de entrada
df = pd.read_csv("evosuitaux.csv")

# Função para gerar os caminhos dos arquivos
def generate_paths(class_name, root_path):
    package_path = class_name.replace('.', '/')
    evosuite_test = f"{root_path}evosuite-tests/{package_path}EvoSuiteTest.java"
    source_file = f"{root_path}src/main/java/{package_path}.java"
    return evosuite_test, source_file

# Aplicar a transformação para gerar os caminhos
df[['file', 'source_file']] = df.apply(lambda row: pd.Series(generate_paths(row['class'], row['root_path'])), axis=1)

# Filtrar para manter apenas as linhas onde o arquivo evosuite existe
df = df[df["file"].apply(os.path.exists)]

# Criar saída `evosuittestsemll_{project}`
for project, df_project in df.groupby("project"):
    output_filename = f"evosuittestsemll_{project}"
    df_project[['project', 'file', 'source_file']].to_csv(output_filename, index=False, header=False)
    print(f"Arquivo gerado: {output_filename}")

# Adicionar colunas fixas
df["method"] = "*"
df["result"] = "SUCCESS"

# Função para obter métricas do Lizard
def analyze_code_metrics(file_path):
    analysis = lizard.analyze_file(file_path)
    return pd.Series([analysis.nloc, analysis.CCN, analysis.token_count, len(analysis.function_list)])

# Função para contar asserts nos métodos
def count_assertions_in_methods(java_file):
    with open(java_file, 'r', encoding='utf-8') as file:
        code = file.read()

    try:
        tree = javalang.parse.parse(code)
    except javalang.parser.JavaSyntaxError:
        return pd.Series([0, 0, 0])  # Retorna 0 caso o código não possa ser analisado

    total_assertions = 0
    methods_without_assertions = 0
    total_methods = 0

    for _, node in tree.filter(javalang.tree.MethodDeclaration):
        total_methods += 1
        assertion_count = sum(1 for _, stmt in node.filter(javalang.tree.MethodInvocation) if stmt.member.startswith("assert") or stmt.member.startswith("fail"))

        total_assertions += assertion_count
        if assertion_count == 0:
            methods_without_assertions += 1

    return pd.Series([total_assertions, methods_without_assertions, total_methods])

# Aplicar as funções de análise
df[['lizard_nloc', 'lizard_ccn', 'lizard_token', 'lizard_function_count']] = df['file'].apply(analyze_code_metrics)
df[['total_assertion', 'methods_without_assertions', 'total_methods']] = df['file'].apply(count_assertions_in_methods)

# Selecionar colunas finais
df_final = df[["project","class", "method", "result", "file", "lizard_nloc", "lizard_ccn", "lizard_token", "lizard_function_count", "total_assertion", "methods_without_assertions", "total_methods"]]

# Criar saída `evosuit_{project}.csv`
for project, df_project in df_final.groupby("project"):
    output_filename = f"evosuit_{project}.csv"
    df_project.drop(columns=["project"], inplace=True)  # Remover a coluna 'project' antes de salvar
    df_project.to_csv(output_filename, index=False)
    print(f"Arquivo gerado: {output_filename}")
