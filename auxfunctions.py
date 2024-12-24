import torch
import argparse
from transformers import pipeline
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer,AutoModelForCausalLM
import shutil
#from optimum.intel.openvino import OVModelForCausalLM
import gc
import os

def load_env_file(file_path):
    env_dict = {}
    try:
        with open(file_path, 'r') as f:
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

    except FileNotFoundError:
        print(f"{file_path} not found.")

    return env_dict

file_path = '.env'
env_data = load_env_file(file_path)

HF_TOKEN = env_data['HF_TOKEN']
if not HF_TOKEN:
    raise ValueError("Hugging Face token (HF_TOKEN) not found in .env file.")

models = {}
tokenizers = {}

file_repo={
    "OpenCoder-8B-Instruct-Q6_K.gguf":"lmstudio-community/OpenCoder-8B-Instruct-GGUF",
    "Yi-Coder-9B-Chat-Q4_K_M.gguf":"lmstudio-community/Yi-Coder-9B-Chat-GGUF",
    "EXAONE-3.5-2.4B-Instruct-BF16.gguf":"LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF",
    "granite-3.1-8b-instruct-Q6_K.gguf":"lmstudio-community/granite-3.1-8b-instruct-GGUF",
    "Llama-3.2-3B-Instruct-f16.gguf":"second-state/Llama-3.2-3B-Instruct-GGUF",
    "gemma-2-9b-it-Q4_K_M-fp16.gguf":"bartowski/gemma-2-9b-it-GGUF",
    "Ministral-8B-Instruct-2410-Q6_K_L.gguf":"bartowski/Ministral-8B-Instruct-2410-GGUF",
    "codegemma-7b-it-Q6_K.gguf":"second-state/CodeGemma-7b-it-GGUF",
    "matteogeniaccio.phi-4.Q3_K_M.gguf":"DevQuasar/matteogeniaccio.phi-4-GGUF",
    "internlm2_5-7b-chat-q8_0.gguf":"internlm/internlm2_5-7b-chat-gguf",
}
# Path where models are stored
MODEL_DIR = env_data.get("model_dir","./models")
os.makedirs(MODEL_DIR, exist_ok=True)

def generate_prompt(messages,model_name):
    prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
    sysmessage = ""
    usermessage = ""
    for messageOb in messages:
        if(messageOb['role']=="system"):
            sysmessage = messageOb['content']
        elif(messageOb['role']=="user"):
            usermessage = messageOb['content']
        else:
            print("error:")
            print(messageOb)
    if(model_name in ["meta-llama/Llama-3.2-1B-Instruct","meta-llama/Llama-3.2-3B-Instruct" ,
                      "OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov"]):
        if(len(sysmessage)>0):
            prompt += f"{sysmessage}<|eot_id|><|eot_id|><|start_header_id|>user<|end_header_id|>{usermessage}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        else:
            prompt = usermessage
    elif(model_name in ["Qwen/Qwen2.5-Coder-0.5B-Instruct","Qwen/Qwen2.5-Coder-1.5B-Instruct","infly/OpenCoder-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct","Salesforce/xLAM-1b-fc-r","ibm-granite/granite-3.1-1b-a400m-instruct",
    "deepseek-ai/deepseek-coder-1.3b-instruct","tiiuae/Falcon3-1B-Instruct","google/gemma2-2b-it"] or 
    model_name.endswith(".gguf")):
        prompt = messages
    else:
        prompt =""
        if(len(sysmessage)>0):
            prompt =f"{sysmessage}\n"
        prompt += f"{usermessage}"
    return prompt,sysmessage,usermessage

def remove_repeated_last_line(text):
    lines = text.splitlines()  # Split the input into lines
    if not lines:
        return text  # If the input is empty, return it as is
    
    last_line = lines[-1]  # Get the last line
    unique_lines = [line for line in lines if line != last_line]  # Remove all repetitions
    unique_lines.append(last_line)  # Keep one instance of the last line

    return "\n".join(unique_lines)  # Reconstruct the string

def filterMessage(messages):
    messagesNew =[]
    for messageOb in messages:
        if(len(messageOb["content"])>0):
            messagesNew.append(messageOb)
    return messagesNew

def generate_model(prompt,model_name,temperature,max_tokens):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    file_name = None
    if model_name.endswith(".gguf"):
        file_name = model_name
        model_name = os.path.join("gguf", model_name)
    if not model_name in models:
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            print("Downloading pretrained model..."+model_name)
            if(file_name):
                download_model(model_name=file_repo[file_name],file=file_name)
            else:
                download_model(model_name=model_name)
        if model_name in ["google/recurrentgemma-2b-it","google/codegemma-2b"
            ,"ibm-granite/granite-3.1-1b-a400m-instruct"]:
            models[model_name] = AutoModelForCausalLM.from_pretrained(model_path,
                                                                      device_map=device
                                                                      )
            tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
        elif model_name.endswith(".gguf"):
            from llama_cpp import Llama
            models[model_name] = Llama(model_path,n_ctx=len(str(prompt))+max_tokens,
                verbose=False, gpu_layers=20)
        elif model_name in ["OpenVINO/codegen25-7b-multi-int4-ov","OpenVINO/codegen25-7b-multi-fp16-ov"]:
            tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            tokenizers[model_name].pad_token = tokenizers[model_name].eos_token
            tokenizers[model_name].add_special_tokens({"pad_token": "<pad>"})
            tokenizers[model_name].padding_side = 'right'
            models[model_name] = OVModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        elif model_name in ["Qwen/Qwen2.5-Coder-0.5B-Instruct","Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct","Salesforce/xLAM-1b-fc-r","infly/OpenCoder-1.5B-Instruct",
        "deepseek-ai/deepseek-coder-1.3b-instruct"
        ]:
            models[model_name] = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map=device,
                trust_remote_code=True
            )
            tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
        elif model_name in ["tiiuae/Falcon3-1B-Instruct","01-ai/Yi-Coder-1.5B","google/gemma2-2b-it"]:
            models[model_name] = pipeline("text-generation",
                                          model=model_path, device_map=device)
        else:
            # Load model and tokenizer
            tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
            tokenizers[model_name].add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
            tokenizers[model_name].padding_side = 'right'
            models[model_name] = pipeline("text-generation",pad_token_id=tokenizers[model_name].pad_token_id
                                          ,tokenizer=tokenizers[model_name],
                                          model=model_path, device_map=device)
    if model_name.endswith(".gguf"):
        return models[model_name].create_chat_completion(
                messages=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )['choices'][0]['message']['content']
    if model_name in ["HuggingFaceTB/SmolLM2-1.7B-Instruct","Salesforce/xLAM-1b-fc-r",
            "deepseek-ai/deepseek-coder-1.3b-instruct","infly/OpenCoder-1.5B-Instruct"]:
        input_text=tokenizers[model_name].apply_chat_template(prompt, tokenize=False)
        inputs = tokenizers[model_name](input_text, return_tensors="pt", padding=True, truncation=True).to(device)
         
        outputs = models[model_name].generate(inputs["input_ids"], max_new_tokens=max_tokens, 
            temperature=temperature, pad_token_id=tokenizers[model_name].pad_token_id, 
                attention_mask=inputs["attention_mask"],eos_token_id=tokenizers[model_name].eos_token_id,
                top_p=0.9, do_sample=True)
        return tokenizers[model_name].decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    if model_name in ["Qwen/Qwen2.5-Coder-0.5B-Instruct","Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "ibm-granite/granite-3.1-1b-a400m-instruct"]:
        text = tokenizers[model_name].apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizers[model_name]([text], return_tensors="pt").to(device)

        generated_ids = models[model_name].generate(
            **model_inputs,
            max_new_tokens=max_tokens,temperature=temperature,
            top_p=0.95, do_sample=True       
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        return tokenizers[model_name].batch_decode(generated_ids, skip_special_tokens=True)[0]  
    if model_name in ["google/recurrentgemma-2b-it","google/codegemma-2b"]:
        return tokenizers[model_name].decode(models[model_name].generate(
            **tokenizers[model_name](prompt, return_tensors="pt").to(device),
            max_new_tokens=max_tokens+len(prompt),eos_token_id=tokenizers[model_name].eos_token_id,
            temperature=temperature, do_sample=False,pad_token_id=tokenizers[model_name].pad_token_id,
            use_cache=False)[0])
    if model_name in ["Intel/Mistral-7B-v0.1-int4-inc","OpenVINO/codegen25-7b-multi-fp16-ov",
                      "OpenVINO/starcoder2-15b-int4-ov","OpenVINO/starcoder2-7b-fp16-ov",
                      "OpenVINO/starcoder2-7b-int8-ov","OpenVINO/codegen25-7b-multi-int4-ov",
                      "OpenVINO/starcoder2-3b-fp16-ov",
                      "OpenVINO/starcoder2-7b-int4-ov","OpenVINO/starcoder2-3b-int4-ov"]:
        return tokenizers[model_name].decode(models[model_name].generate(
            **tokenizers[model_name](prompt, return_tensors="pt").to(device),do_sample=True,
            pad_token_id=tokenizers[model_name].pad_token_id,max_new_tokens=max_tokens,
            eos_token_id=tokenizers[model_name].eos_token_id, temperature=temperature)[0])
    return models[model_name](prompt,temperature=temperature,max_new_tokens=max_tokens,return_full_text=False,do_sample=True)[0]['generated_text']

# 1. List available text generation models from Hugging Face Hub
def list_hf_models():
    api = HfApi(token=HF_TOKEN)
    models = api.list_models(filter="text-generation")
    return [model.modelId for model in models]

# 2. Download model from Hugging Face Hub
def download_model(model_name,file=None):
    model_path = os.path.join(MODEL_DIR, model_name)
    if os.path.exists(model_path):
        return f"Model '{model_name}' is already downloaded."

    try:
        if(file):
            local_cache = os.path.join(MODEL_DIR,'gguf',".cache")
            snapshot_download(repo_id=model_name, local_dir=os.path.join(MODEL_DIR, 'gguf'), token=HF_TOKEN, 
                allow_patterns=[file],cache_dir=local_cache)
            if(os.path.exists(local_cache)):
                print(local_cache)
                shutil.rmtree(local_cache)
        else:
            local_cache = os.path.join(model_path,".cache")
            snapshot_download(repo_id=model_name, local_dir=model_path,cache_dir=local_cache,
            token=HF_TOKEN,ignore_patterns=["*onnx*","runs","*guff*"])
            if(os.path.exists(local_cache)):
                shutil.rmtree(local_cache)

        return f"Model '{model_name}' downloaded successfully."
       
    except Exception as e:
        raise ValueError(f"Failed to download model '{model_name}': {str(e)}")

def clear_models_from_mem():
    # Clear all models in the dictionary
    for key in list(models.keys()):
        del models[key]

    # Clear the dictionary itself
    models.clear()

    # If using PyTorch, free up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run garbage collection to free up memory
    gc.collect()