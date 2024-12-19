import os

from loadenv import load_env_file
import requests
import torch
import argparse
from transformers import pipeline
from flask import Flask, jsonify, request
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer,AutoModelForCausalLM
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
# Path where models are stored
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

    

def generate_model(prompt,model_name,temperature,max_tokens):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not model_name in models:
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            try:
                print("Downloading pretrained model..."+model_name)
                download_model(model_name)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
        if model_name in ["google/recurrentgemma-2b-it","google/codegemma-2b"]:
            models[model_name] = AutoModelForCausalLM.from_pretrained(model_path,
                                                                      device_map=device
                                                                      )
            tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
        elif model_name in ["Qwen/Qwen2.5-Coder-0.5B-Instruct","Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct","Salesforce/xLAM-1b-fc-r",
        "deepseek-ai/deepseek-coder-1.3b-instruct"]:
            models[model_name] = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map=device,
                trust_remote_code=True
            )
            tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        else:
            # Load model and tokenizer
            tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
            tokenizers[model_name].add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
            tokenizers[model_name].padding_side = 'right'
            models[model_name] = pipeline("text-generation",pad_token_id=tokenizers[model_name].pad_token_id
                                          ,tokenizer=tokenizers[model_name],
                                          model=model_path, device_map=device)
    if model_name in ["HuggingFaceTB/SmolLM2-1.7B-Instruct","Salesforce/xLAM-1b-fc-r",
            "deepseek-ai/deepseek-coder-1.3b-instruct"]:
        input_text=tokenizers[model_name].apply_chat_template(prompt, tokenize=False)
        inputs = tokenizers[model_name](input_text, return_tensors="pt", padding=True, truncation=True).to(device)
         
        outputs = models[model_name].generate(inputs["input_ids"], max_new_tokens=max_tokens, 
            temperature=temperature, pad_token_id=tokenizers[model_name].pad_token_id, 
                attention_mask=inputs["attention_mask"],eos_token_id=tokenizers[model_name].eos_token_id,
                top_p=0.9, do_sample=True)
        return tokenizers[model_name].decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    if model_name in ["Qwen/Qwen2.5-Coder-0.5B-Instruct","Qwen/Qwen2.5-Coder-1.5B-Instruct"]:
        text = tokenizers[model_name].apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizers[model_name]([text], return_tensors="pt").to(device)

        generated_ids = models[model_name].generate(
            **model_inputs,
            max_new_tokens=max_tokens,temperature=temperature,
            top_p=0.9, do_sample=True       
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
            **tokenizers[model_name](prompt, return_tensors="pt").to(device),do_sample=False,
            pad_token_id=tokenizers[model_name].pad_token_id,max_new_tokens=max_tokens,
            eos_token_id=tokenizers[model_name].eos_token_id, temperature=temperature)[0])
    return models[model_name](prompt,temperature=temperature,max_new_tokens=max_tokens,return_full_text=False)[0]['generated_text']

# 1. List available text generation models from Hugging Face Hub
def list_hf_models():
    api = HfApi(token=HF_TOKEN)
    models = api.list_models(filter="text-generation")
    return [model.modelId for model in models]


# 2. Download model from Hugging Face Hub
def download_model(model_name):
    model_path = os.path.join(MODEL_DIR, model_name)
    if os.path.exists(model_path):
        return f"Model '{model_name}' is already downloaded."

    try:
        patterns = [
            "config.json",  # Model configuration
            "pytorch_model.bin",  # Model weights
            "tokenizer.json",  # Tokenizer
            "vocab.json",  # Tokenizer vocabulary (if applicable)
            "merges.txt",  # Tokenizer merges (if applicable)
            "model.safetensors",
            "generation_config.json",
            "tokenizer_config.json", 
        ]
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


@app.route('/generateChatTester', methods=['POST','GET'])
def generate_text_GPT():
    data = request.get_json()

    # Extract parameters
    model_name = data.get('local_model','meta-llama/Llama-3.2-1B-Instruct')
    #meta-llama/Llama-3.2-1B-Instruct
    #Qwen/Qwen2.5-Coder-1.5B-Instruct
    #HuggingFaceTB/SmolLM2-1.7B-Instruct
    #Salesforce/xLAM-1b-fc-r
    #deepseek-ai/deepseek-coder-1.3b-instruct
    
    messages = data.get('messages')
    max_tokens = data.get('max_tokens', 512)
    print("maxtokens:"+str(max_tokens))
    temperature = data.get('temperature', 0.7)
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
        prompt += f"{sysmessage}<|eot_id|><|eot_id|><|start_header_id|>user<|end_header_id|>{usermessage}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    elif(model_name in ["Qwen/Qwen2.5-Coder-0.5B-Instruct","Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct","Salesforce/xLAM-1b-fc-r","deepseek-ai/deepseek-coder-1.3b-instruct"]):
        prompt = messages
    else:
        prompt = f"{sysmessage}\n{usermessage}"
    if not model_name or not prompt:
        return jsonify({"error": "'model' and 'prompt' are required."}), 400
    #print("prompt:"+prompt)
    output = generate_model(prompt=prompt,model_name=model_name,
        temperature=temperature,max_tokens=max_tokens)
    #print("rawresponse:" + output)
    # Split the output from the superprompt length
    if(model_name in [""]):
        assistant_response = output[len(prompt):].strip()
    elif(model_name in ["Qwen/Qwen2.5-Coder-0.5B-Instruct"
            ,"Qwen/Qwen2.5-Coder-1.5B-Instruct"]):
        assistant_response = output.replace("response:","",1)
        prompt = f"{sysmessage}\n{usermessage}"
    elif(model_name in ["HuggingFaceTB/SmolLM2-1.7B-Instruct","Salesforce/xLAM-1b-fc-r",
        "deepseek-ai/deepseek-coder-1.3b-instruct"]):
        assistant_response = output
        prompt = f"{sysmessage}\n{usermessage}"
    else:
        assistant_response = output
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
    
# Replace with Gemini's API URL and API key
GEMINI_API_URL = "https://api.gemini.example/v1/query"
GEMINI_API_KEY = env_data["GEMINI_TOKEN"]

@app.route("/openai", methods=["POST"])
def openai_to_gemini():
    try:
        # Get the OpenAI-style input
        openai_request = request.json
        if not openai_request:
            return jsonify({"error": "Invalid input"}), 400

        # Transform the OpenAI request to Gemini request
        gemini_payload = {
            "query": openai_request.get("messages", [])[-1]["content"],  # Assuming the last message contains the user query
            "temperature": openai_request.get("temperature", 1.0),
            "max_tokens": openai_request.get("max_tokens", 150)
        }

        # Send the request to Gemini API
        gemini_response = requests.post(
            GEMINI_API_URL,
            headers={"Authorization": f"Bearer {GEMINI_API_KEY}"},
            json=gemini_payload
        )

        if gemini_response.status_code != 200:
            return jsonify({"error": "Failed to query Gemini", "details": gemini_response.text}), 500

        # Transform Gemini response back to OpenAI format
        gemini_data = gemini_response.json()
        openai_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": gemini_data.get("response", "")
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ],
            "usage": {
                "prompt_tokens": gemini_data.get("prompt_tokens", 0),
                "completion_tokens": gemini_data.get("completion_tokens", 0),
                "total_tokens": gemini_data.get("total_tokens", 0)
            }
        }

        return jsonify(openai_response)

    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VLLM Local Server")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address for the server')
    parser.add_argument('--port', type=int, default=5000, help='Port number for the server')
    args = parser.parse_args()

    app.run(host=args.host, port=args.port)
