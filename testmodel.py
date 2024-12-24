import os
from main import generate_model,clear_models
file_path = '.env'
from loadenv import load_env_file
env_data = load_env_file(file_path)
# Path to the folder containing GGUF model files
models_folder = os.path.join(env_data.get("model_dir","./models"),"gguf")
models=["meta-llama/Llama-3.2-1B-Instruct"]
prompt="give me e small java class that calculate the area of a polygon"
messages=[
    {
        "role": "system",
        "content": "You are a assitant that gives pure code as aswer"
    }
    ,
    {
        "role": "user",
        "content": prompt
    }
]
# Iterate through all files in the folder
for model_name in models:
    print(f"Testing model: {model_name}")

    try:
        prompt,sysmessage,usermessage = generate_prompt(messages=messages,model_name=model_name)

        # Create chat completion
        result = generate_model(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=256,
            temperature=0.7
        )

        # Print the output
        print(f"Response from {model_name}:")
        print(result)
        clear_models()
    except Exception as e:
        print(f"An error occurred while testing {model_name}: {e}")
        print(e)
