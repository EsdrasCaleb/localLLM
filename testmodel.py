import os
from llama_cpp import Llama
file_path = '.env'
from loadenv import load_env_file
env_data = load_env_file(file_path)
# Path to the folder containing GGUF model files
models_folder = os.path.join(env_data.get("model_dir","./models"),"gguf")
prompt="// Focal class\npublic class Query {\nprotected String serverURL,\n    associatesID,\n    token = \"DSB0XDDW1GQ3S\", //don't change A4J won't work without this. Used for tracking tool use.\n    searchType,\n    type,\n    page,\n    offer;\nprotected ArrayList searchValues;\na4jUtil jawsUtil = new a4jUtil();\npublic Query(){}\n// Focal method\npublic String queryGenerator(String searchType, String type, String page, String offer, ArrayList searchValues) {\n        //    log.debug(\"queryGenerator - in\");\n        StringBuffer buffer = new StringBuffer();\n        buffer.append(serverURL);\n        buffer.append(\"?\");\n        buffer.append(\"t=\");\n        buffer.append(associatesID);\n        buffer.append(\"&\");\n        buffer.append(\"dev-t=\");\n        buffer.append(token);\n        buffer.append(\"&\");\n        buffer.append(searchType);\n        buffer.append(\"=\");\n        buffer.append(generateMultipleSearchString(searchType, searchValues));\n        buffer.append(\"&\");\n        buffer.append(\"type=\");\n        buffer.append(type);\n        buffer.append(\"&\");\n        buffer.append(\"offerpage=\");\n        buffer.append(page);\n        buffer.append(\"&\");\n        buffer.append(\"offer=\");\n        buffer.append(offer);\n        buffer.append(\"&\");\n        buffer.append(\"f=xml\");\n        //      log.debug(\"queryGenerator - out\");\n        return new String(buffer);\n    }\n}\nPlease infer the intention of the \"queryGenerator(String, String, String, String, ArrayList)\".\n"
prompt="give me e small java class that calculate the area of a polygon"

# Iterate through all files in the folder
for model_filename in os.listdir(models_folder):
    model_path = os.path.join(models_folder, model_filename)
    # Check if the file is a GGUF model
    #if model_filename.startswith("ggml-c4ai-command-r7b-12-2024-q4_k") and model_filename.endswith(".gguf"):
    if model_filename in ["Yi-Coder-9B-Chat-Q4_K_M.gguf","Ministral-8B-Instruct-2410-Q6_K_L.gguf"]:
        print(f"Testing model: {model_filename}")

        try:
            # Load the model
            llm = Llama(model_path=model_path, verbose=False, gpu_layers=21)

            # Create chat completion
            response = llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9
            )

            # Print the output
            print(f"Response from {model_filename}:")
            print(response['choices'][0]['message']['content'])
        except Exception as e:
            print(f"An error occurred while testing {model_filename}: {e}")
            print(e)
