import os
from auxfunctions import generate_model,generate_prompt,load_env_file,clear_models_from_mem
file_path = '.env'
env_data = load_env_file(file_path)
# Path to the folder containing GGUF model files
models_folder = os.path.join(env_data.get("model_dir","./models"),"gguf")
models=["EXAONE-3.5-2.4B-Instruct-BF16.gguf"]
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

    
    prompt,sysmessage,usermessage = generate_prompt(messages=messages,model_name=model_name)

    # Create chat completion
    result = generate_model(
        prompt=prompt,
        model_name=model_name,
        max_tokens=256,
        temperature=0.7
    )

    # Print the output
    print(f"Response from {model_name}:")
    print(result)
    clear_models_from_mem()
    
