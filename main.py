import os

from loadenv import load_env_file
import requests
import torch
import argparse
from transformers import pipeline
from flask import Flask, jsonify, request
from huggingface_hub import HfApi, snapshot_download
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import openvino_genai
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
                download_model(model_name)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
        if model_name == "OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov":
            # Set up the main and draft models
            main_device = "CPU"
            draft_device = "CPU"
            draft_model_path = os.path.join(MODEL_DIR, "meta-llama/Llama-3.2-8B-Instruct")
            draft_model = openvino_genai.draft_model(draft_model_path, draft_device)

            scheduler_config = openvino_genai.SchedulerConfig()
            scheduler_config.cache_size = 4

            # Initialize the LLM pipeline with the models and configuration
            models[model_name] = openvino_genai.LLMPipeline(
                model_path,
                main_device,
                scheduler_config=scheduler_config,
                draft_model=draft_model
            )
        elif model_name == "google/recurrentgemma-2b-it":
            models[model_name] = AutoModelForCausalLM.from_pretrained(model_path,
                                                                      device_map=device
                                                                      )
            tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
        elif model_name == "Intel/Mistral-7B-v0.1-int4-inc":
            models[model_name] = AutoModelForCausalLM.from_pretrained(model_path,
                                                         device_map=device,
                                                         trust_remote_code=True,
                                                         use_neural_speed=False,
                                                         )
            tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        else:
            models[model_name] = pipeline("text-generation",model=model_path, device_map=device)
    if model_name == "OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov":
        # Create a GenerationConfig object and set the parameters
        config = openvino_genai.GenerationConfig()
        config.num_assistant_tokens = 3
        config.max_new_tokens = max_tokens
        config.temperature = temperature  # Adding temperature control
        return models[model_name].generate(prompt,config)
    if model_name == "google/recurrentgemma-2b-it":
        return tokenizers[model_name].decode(models[model_name].generate(
            **tokenizers[model_name](prompt, return_tensors="pt").to(device),
            max_new_tokens=max_tokens, temperature=temperature)[0])
    if model_name == "Intel/Mistral-7B-v0.1-int4-inc":
        return tokenizers[model_name].decode(models[model_name].generate(
            **tokenizers[model_name](prompt, return_tensors="pt").to(device),
            max_new_tokens=max_tokens+len(prompt), temperature=temperature)[0])
    return models[model_name](prompt,temperature=temperature,max_new_tokens=max_tokens+len(prompt))[0]['generated_text']

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
    model_name = "Intel/Mistral-7B-v0.1-int4-inc"
    #google/recurrentgemma-2b-it
    #OpenVINO/starcoder2-7b-int4-ov
    #OpenVINO/codegen25-7b-multi-int4-ov
    #nvidia/Hymba-1.5B-Instruct
    #HuggingFaceTB/SmolLM2-1.7B-Instruct
    #meta-llama/Llama-3.2-1B-Instruct
    #DistilLLaMA-1.3B
    #Intel/Mistral-7B-v0.1-int4-inc
    #OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov
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
    if(model_name == "meta-llama/Llama-3.2-1B-Instruct" or model_name=="OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov"):
        prompt += f"{sysmessage}<|eot_id|><|eot_id|><|start_header_id|>user<|end_header_id|>{usermessage}<|eot_id|>"
    else:
        prompt = f"Context:{sysmessage}\nUser:{usermessage}"
    if not model_name or not prompt:
        return jsonify({"error": "'model' and 'prompt' are required."}), 400

    output = generate_model(prompt=prompt,model_name=model_name,
        temperature=temperature,max_tokens=max_tokens)

    # Split the output from the superprompt length
    if(model_name != "OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov" and model_name!="Intel/Mistral-7B-v0.1-int4-inc"):
        assistant_response = output[len(prompt):].strip()
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
