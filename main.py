import os
import torch
import argparse
from flask import Flask, jsonify, request
from vllm import LLM,SamplingParams
from huggingface_hub import HfApi, snapshot_download
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Hugging Face token (HF_TOKEN) not found in .env file.")

app = Flask(__name__)
models = {}
# Path where models are stored
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

def get_model(model_name):
    if not model_name in models:
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            try:
                download_model(model_name)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models[model_name] = = LLM(model=model_path, device=device)
        params = SamplingParams(temperature=temperature,max_tokens=max_tokens)
    return models[model_name]

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


@app.route('/download_model', methods=['POST'])
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
    llm = get_model(model_name)
    try:
        
        output = llm.generate(
            prompt,
            params
        )
        return jsonify({
            "model": model_name,
            "prompt": prompt,
            "choices": [{"text": output}],
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
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    messages = data.get('messages')
    max_tokens = data.get('max_tokens', 512)
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
    prompt += f"{sysmessage}<|eot_id|><|eot_id|><|start_header_id|>user<|end_header_id|>{usermessage}" 
    if not model_name or not prompt:
        return jsonify({"error": "'model' and 'prompt' are required."}), 400

    llm = get_model(model_name)
    output = llm.generate(
        prompt,
        params
    )
    return jsonify({
        "model": model_name,
        "prompt": prompt,
        "choices": [{"text": output}],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(output.split()),
            "total_tokens": len(prompt.split()) + len(output.split())
        }
    })
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VLLM Local Server")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address for the server')
    parser.add_argument('--port', type=int, default=5000, help='Port number for the server')
    args = parser.parse_args()

    app.run(host=args.host, port=args.port)
