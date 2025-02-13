auxarray = {
"com.ib.client.UnderComp":"../SF110/1_tullibee/src/main/java/com/ib/client/UnderComp.java",
"com.ib.client.TickType":"../SF110/1_tullibee/src/main/java/com/ib/client/TickType.java",
"com.ib.client.EWrapperMsgGenerator":"../SF110/1_tullibee/src/main/java/com/ib/client/EWrapperMsgGenerator.java",
"com.ib.client.TagValue":"../SF110/1_tullibee/src/main/java/com/ib/client/TagValue.java",
"com.ib.client.ScannerSubscription":"../SF110/1_tullibee/src/main/java/com/ib/client/ScannerSubscription.java",
"com.ib.client.EClientSocket":"../SF110/1_tullibee/src/main/java/com/ib/client/EClientSocket.java",
"com.ib.client.ComboLeg":"../SF110/1_tullibee/src/main/java/com/ib/client/ComboLeg.java",
"com.ib.client.Util":"../SF110/1_tullibee/src/main/java/com/ib/client/Util.java",
"com.ib.client.Contract":"../SF110/1_tullibee/src/main/java/com/ib/client/Contract.java",
"com.ib.client.Execution":"../SF110/1_tullibee/src/main/java/com/ib/client/Execution.java",
"com.ib.client.ExecutionFilter":"../SF110/1_tullibee/src/main/java/com/ib/client/ExecutionFilter.java",
"com.ib.client.EReader":"../SF110/1_tullibee/src/main/java/com/ib/client/EReader.java",
"com.ib.client.Order":"../SF110/1_tullibee/src/main/java/com/ib/client/Order.java",
"com.ib.client.OrderState":"../SF110/1_tullibee/src/main/java/com/ib/client/OrderState.java",
"com.ib.client.AnyWrapperMsgGenerator":"../SF110/1_tullibee/src/main/java/com/ib/client/AnyWrapperMsgGenerator.java",
}

classes = auxarray.keys()

import os
import csv

# Base directory for file paths
base_dir = "../SF110/1_tullibee/evosuite-tests"

# Output CSV file
output_csv = "output.csv"

# Prepare data for CSV
csv_data = []

for cls in classes:
    # Convert class to file path
    class_parts = cls.split(".")
    class_name = class_parts[-1]
    directory_path = "/".join(class_parts[:-1])
    file_path = f"{base_dir}/{directory_path}/{class_name}EvoSuiteTest.java"

    if os.path.exists(file_path):
        csv_data.append([cls, file_path])
    else:
        print(f"Error: File not found for class '{cls}' at path '{file_path}'")

# Write to CSV if there is data
if csv_data:
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Class", "File Path"])
        writer.writerows(csv_data)
    print(f"CSV file '{output_csv}' generated successfully.")
else:
    print("No valid files found. CSV not generated.")