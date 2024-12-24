import os

from loadenv import load_env_file
import requests
import torch
import argparse
from transformers import pipeline
from flask import Flask, jsonify, request
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer,AutoModelForCausalLM
#from optimum.intel.openvino import OVModelForCausalLM
import gc
#from dotenv import load_dotenv
# Example usage
file_path = '.env'
env_data = load_env_file(file_path)
# Load environment variables
#load_dotenv()
HF_TOKEN = env_data['HF_TOKEN']
if not HF_TOKEN:
    raise ValueError("Hugging Face token (HF_TOKEN) not found in .env file.")

app = Flask(__name__)
models = {}
tokenizers = {}

file_repo={
    "OpenCoder-8B-Instruct-Q6_K.gguf":"lmstudio-community/OpenCoder-8B-Instruct-GGUF",
    "Yi-Coder-9B-Chat-Q4_K_M.gguf":"lmstudio-community/Yi-Coder-9B-Chat-GGUF",
    "EXAONE-3.5-2.4B-Instruct-BF16.ggf":"LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF",
    "granite-3.1-8b-instruct-Q6_K.gguf":"lmstudio-community/granite-3.1-8b-instruct-GGUF",
    "Llama-3.2-3B-Instruct-f16.gguf":"second-state/Llama-3.2-3B-Instruct-GGUF",
    "gemma-2-9b-it-Q4_K_M-fp16.gguf":"bartowski/gemma-2-9b-it-GGUF",
    "Ministral-8B-Instruct-2410-Q6_K_L.gguf":"bartowski/Ministral-8B-Instruct-2410-GGUF",
    "codegemma-7b-it-Q6_K.gguf":"second-state/CodeGemma-7b-it-GGUF",
    "matteogeniaccio.phi-4.Q3_K_M.gguf":"matteogeniaccio/phi-4",
    "internlm2_5-7b-chat-q8_0.gguf":"internlm/internlm2_5-7b-chat-gguf",
}
# Path where models are stored
MODEL_DIR = env_data.get("model_dir","./models")
os.makedirs(MODEL_DIR, exist_ok=True)

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
            try:
                print("Downloading pretrained model..."+model_name)
                if(file_name):
                    download_model(model_name=file_repo[model_name],file=file_name)
                else:
                    download_model(model_name=model_name)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
        if model_name in ["google/recurrentgemma-2b-it","google/codegemma-2b"
            ,"ibm-granite/granite-3.1-1b-a400m-instruct"]:
            models[model_name] = AutoModelForCausalLM.from_pretrained(model_path,
                                                                      device_map=device
                                                                      )
            tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
        elif model_name.endswith(".gguf"):
            from llama_cpp import Llama
            print(f"\n\n\n\n{model_path}\n\n\n\n")
            models[model_name] = Llama(model_path,
            _ctx=len(str(prompt))+max_tokens,verbose=False, gpu_layers=20)
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
            snapshot_download(repo_id=model_name, local_dir=os.path.join(MODEL_DIR, 'gguf'), token=HF_TOKEN, 
                allow_patterns=[file])
        else:
            snapshot_download(repo_id=model_name, local_dir=model_path, token=HF_TOKEN)
        return f"Model '{model_name}' downloaded successfully."
    except Exception as e:
        raise ValueError(f"Failed to download model '{model_name}': {str(e)}")


# Flask API endpoints

@app.before_request
def log_request_info():
    print(f"Request received: {request.method} {request.url}")
    if request.data:
        print(f"Payload: {request.data.decode('utf-8')}")

# CLear model from Hugging Face Hub on memory
@app.route('/clear_models', methods=['GET'])
def clear_models():
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

@app.route('/list_models', methods=['GET'])
def list_models_endpoint():
    return jsonify(list_hf_models())


@app.route('/download_model', methods=['POST','GET'])
def download_model_endpoint():
    data = request.get_json()
    model_name = data.get('model_name')
    if not model_name:
        return jsonify({"error": "'model_name' is required."}), 400

    try:
        result = download_model(model_name)
        return jsonify({"message": result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()

    # Extract parameters
    model_name = data.get('model')
    prompt = data.get('prompt')
    max_tokens = data.get('max_tokens', 512)
    temperature = data.get('temperature', 0.7)

    if not model_name or not prompt:
        return jsonify({"error": "'model' and 'prompt' are required."}), 400
   
    try:
        output = generate_model(prompt=prompt,model_name=model_name,
            temperature=temperature,max_tokens=max_tokens)

        # Split the output from the superprompt length
        assistant_response = output[len(prompt):].strip()
        return jsonify({
            "model": model_name,
            "prompt": prompt,
            "choices": [{"text": assistant_response}],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(output.split()),
                "total_tokens": len(prompt.split()) + len(output.split())
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate_model', methods=['POST','GET'])
def generate_text_GPT():
    data = request.get_json()

    # Extract parameters
    model_name = data.get('model','Phi-3.5-mini-instruct-Q8_0.gguf')
    
    messages = filterMessage(data.get('messages'))
    max_tokens = data.get('max_tokens', 512)
    print("maxtokens:"+str(max_tokens))
    temperature = data.get('temperature', 0.7)
    prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
    sysmessage = ""
    usermessage = ""
    print(messages)
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
    if not model_name or not prompt:
        return jsonify({"error": "'model' and 'prompt' are required."}), 400
    output = generate_model(prompt=prompt,model_name=model_name,
        temperature=temperature,max_tokens=max_tokens)
    #print("rawresponse:" + output)
    # Split the output from the superprompt length
    if(model_name in ["starcoder2-3b-Q8_0.gguf"]):
        assistant_response = output.replace()[len(prompt):].strip()
    elif(model_name in ["Qwen/Qwen2.5-Coder-0.5B-Instruct"
            ,"Qwen/Qwen2.5-Coder-1.5B-Instruct"]):
        assistant_response = output.replace("response:","",1)
        prompt = f"{sysmessage}\n{usermessage}"
    elif(model_name in ["HuggingFaceTB/SmolLM2-1.7B-Instruct","Salesforce/xLAM-1b-fc-r",
    "infly/OpenCoder-1.5B-Instruct","google/gemma2-2b-it",
    "deepseek-ai/deepseek-coder-1.3b-instruct","tiiuae/Falcon3-1B-Instruct",
    "ibm-granite/granite-3.1-1b-a400m-instruct"] 
    or model_name.endswith(".gguf")):
        assistant_response = output
        prompt = f"{sysmessage}\n{usermessage}"
    else:
        assistant_response = output
    #cleaned_response = remove_repeated_last_line(assistant_response).strip()
    print("prompt:" + prompt+"\n\n\n\n")
    print("response:" + assistant_response)

    return jsonify({
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": assistant_response
                },
                "finish_reason": "stop",
                "index": 0
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(output.split()),
            "total_tokens": len(prompt.split()) + len(output.split())
        }
    })
    

gemini_keys = env_data["g_tokens"].split(",")
gemini_index = 0

@app.route("/google", methods=["POST","GET"])
def openai_to_gemini():
    global gemini_index
    key = gemini_keys[gemini_index]
    gemini_index = (1+gemini_index)%(len(gemini_keys))
    try:
        # Get the OpenAI-style input
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid input"}), 400
        model = data.get('model','gemini-1.5-flash') #gemma-7b-it gemini-2.0-flash-exp
    
        messages = filterMessage(data.get('messages'))
        max_tokens = data.get('max_tokens', 512)
        print("maxtokens:"+str(max_tokens))
        temperature = data.get('temperature', 0.7)
        sysmessage = ""
        usermessage = ""
        for messageOb in messages:
            messagesNew.append({"role": "user", "parts": [{"text": messageOb['content']}]})

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"  # Gemini API endpoint

        data = {
            "contents": messagesNew,
            "generation_config": {
                "temperature": temperature,
                "max_output_tokens": max_tokens  # Correct parameter name for Gemini
            }
        }

        print(data)
        # Send the request to Gemini API
        gemini_response = requests.post(
            url,
            headers={ "Content-Type": "application/json","x-goog-api-key": key},
            json=data
        )

        if gemini_response.status_code != 200:
            print(gemini_response.text)
            return jsonify({"error": "Failed to query Gemini", "details": gemini_response.text}), 500
        
        # Transform Gemini response back to OpenAI format
        response_json = gemini_response.json()
        response = ""
        
        if response_json and 'candidates' in response_json and len(response_json['candidates']) > 0:
          if response_json['candidates'][0] and 'content' in response_json['candidates'][0] and response_json['candidates'][0]['content'] and 'parts' in response_json['candidates'][0]['content']:
              if response_json['candidates'][0]['content']['parts'] and len(response_json['candidates'][0]['content']['parts']) > 0 :
                response = response_json['candidates'][0]['content']['parts'][0]['text']
        openai_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ],
            "usage": {
                "prompt_tokens": response_json["usageMetadata"]["promptTokenCount"],
                "completion_tokens": response_json["usageMetadata"]["candidatesTokenCount"],
                "total_tokens": response_json["usageMetadata"]["totalTokenCount"]
            }
        }

        return jsonify(openai_response)
    except Exception as e:
        print(e)
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

mistral_key = env_data["MISTRAL_API_KEY"]

@app.route("/mistral", methods=["POST","GET"])
def mistral_to_openai():
    from mistralai import Mistral
    try:
        # Get the OpenAI-style input
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid input"}), 400
        model = data.get('model','open-codestral-mamba')
    
        messages = filterMessage(data.get('messages'))

        max_tokens = data.get('max_tokens', 512)
        temperature = data.get('temperature', 0.7)
        client = Mistral(api_key=mistral_key)
        chat_response = client.chat.complete(
            model= model,
            max_tokens= max_tokens,
            temperature= temperature,
            messages = messages
        )
        openai_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": chat_response.choices[0].message.content
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ],
            "usage": {
                "prompt_tokens": chat_response.usage.prompt_tokens,
                "completion_tokens": chat_response.usage.completion_tokens,
                "total_tokens": chat_response.usage.total_tokens,
            }
        }
        return jsonify(openai_response)
    except Exception as e:
        print(e)
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

grok_key =  env_data["grok_key"]
@app.route("/grok", methods=["POST","GET"])
def grok_to_openai():
    try:
        data = request.get_json()
        url = "https://api.x.ai/v1/chat/completions"  # Gemini API endpoint
        model = data.get('model','grok-2-1212')
        
        messages = filterMessage(data.get('messages'))
        max_tokens = data.get('max_tokens', 512)
        temperature = data.get('temperature', 0.7)
        data = {
            "messages":messages,
            "max_tokens":max_tokens,
            "temperature":temperature,
            "stream":False,
            "model":model
        }
        
        # Send the request to Grok
        response = requests.post(
            url,
            headers={ "Content-Type": "application/json","Authorization": f"Bearer {grok_key}"},
            json=data
        )
        if response.status_code != 200:
            print(response.text)
            return jsonify({"error": "Failed to query Gemini", "details": response.text}), 500
        
        return response.json()
    except Exception as e:
        print(e)
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VLLM Local Server")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address for the server')
    parser.add_argument('--port', type=int, default=5000, help='Port number for the server')
    args = parser.parse_args()

    app.run(host=args.host, port=args.port)
