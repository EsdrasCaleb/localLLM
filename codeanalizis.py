import os
import subprocess
import pandas as pd

# Paths to tools and configuration
PMD_PATH = "pmd-bin/bin/pmd"  # Path to PMD executable
RULESET_PATH = "ruleset.xml"  # Path to PMD ruleset
CODE_PATH = "/path/to/your/java/code"  # Path to the Java code
PMD_OUTPUT_CSV = "pmd_output.csv"  # Output CSV for PMD
SONAR_OUTPUT_CSV = "sonar_output.csv"  # Output CSV for SonarQube
FINAL_CSV = "final_report.csv"  # Final merged report

def run_pmd():
    """Run PMD to analyze code and output results in CSV format."""
    print("Running PMD...")
    pmd_command = [
        PMD_PATH,
        "-d", CODE_PATH,
        "-R", RULESET_PATH,
        "-f", "csv",
        "-r", PMD_OUTPUT_CSV
    ]
    subprocess.run(pmd_command, check=True)
    print(f"PMD output saved to {PMD_OUTPUT_CSV}")

def run_sonarqube():
    """Run SonarQube scanner and extract data."""
    print("Running SonarQube...")
    sonar_command = [
        "sonar-scanner",
        f"-Dsonar.projectBaseDir={CODE_PATH}",
    ]
    subprocess.run(sonar_command, check=True)

    # Assuming SonarCSV plugin is used to generate output in CSV
    # Replace with your logic to fetch results as needed
    print(f"SonarQube output should be available via plugin or API.")

def merge_results():
    """Merge PMD and SonarQube outputs into a final CSV report."""
    print("Merging results...")

    # Load PMD output
    pmd_df = pd.read_csv(PMD_OUTPUT_CSV)
    pmd_df = pmd_df.rename(columns={"File": "Filename", "CyclomaticComplexity": "Cyclomatic Complexity"})

    # Load SonarQube output
    sonar_df = pd.read_csv(SONAR_OUTPUT_CSV)
    sonar_df = sonar_df.rename(columns={"file": "Filename", "smells": "Number of Smells"})

    # Merge results
    merged_df = pd.merge(pmd_df, sonar_df, on="Filename", how="outer")
    merged_df.to_csv(FINAL_CSV, index=False)

    print(f"Final report saved to {FINAL_CSV}")

if __name__ == "__main__":
    try:
        run_pmd()
        run_sonarqube()
        merge_results()
    except Exception as e:
        print(f"Error: {e}")
