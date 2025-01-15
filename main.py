import os
import auxfunctions
import requests
import argparse
from flask import Flask, jsonify, request

# Load environment variables
#load_dotenv()
file_path = '.env'
env_data = auxfunctions.load_env_file(file_path)

app = Flask(__name__)

# Flask API endpoints

@app.before_request
def log_request_info():
    print(f"Request received: {request.method} {request.url}")
    if request.data:
        print(f"Payload: {request.data.decode('utf-8')}")

# CLear model from Hugging Face Hub on memory
@app.route('/clear_models', methods=['GET'])
def clear_models():
    auxfunctions.clear_models_from_mem()
    return jsonify({"status":"ok","message":"models cleared"})

@app.route('/list_models', methods=['GET'])
def list_models_endpoint():
    return jsonify(auxfunctions.list_hf_models())


@app.route('/download_model', methods=['POST','GET'])
def download_model_endpoint():
    data = request.get_json()
    model_name = data.get('model_name')
    if not model_name:
        return jsonify({"error": "'model_name' is required."}), 400

    try:
        result = auxfunctions.download_model(model_name)
        return jsonify({"message": result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route('/generate_model', methods=['POST','GET'])
def generate_text():
    data = request.get_json()

    # Extract parameters
    model_name = data.get('model','Phi-3.5-mini-instruct-Q8_0.gguf')
    
    messages = auxfunctions.filterMessage(data.get('messages'))
    max_tokens = data.get('max_tokens', 512)
    print("maxtokens:"+str(max_tokens))
    temperature = data.get('temperature', 0.7)
    prompt,sysmessage,usermessage = auxfunctions.generate_prompt(messages=messages,model_name=model_name)
    if not model_name or not prompt:
        return jsonify({"error": "'model' and 'prompt' are required."}), 400
    try:
        output = auxfunctions.generate_model(prompt=prompt, model_name=model_name,
                                             temperature=temperature, max_tokens=max_tokens)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
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
    
        messages = auxfunctions.filterMessage(data.get('messages'))
        max_tokens = data.get('max_tokens', 512)
        print("maxtokens:"+str(max_tokens))
        temperature = data.get('temperature', 0.7)
        sysmessage = ""
        usermessage = ""
        messagesNew = []
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
    
        messages = auxfunctions.filterMessage(data.get('messages'))

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


@app.route("/hugging", methods=["POST", "GET"])
def hugging_to_openai():
    from huggingface_hub import InferenceClient
    try:
        # Get the OpenAI-style input
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid input"}), 400
        model = data.get('model', 'open-codestral-mamba')

        messages = auxfunctions.filterMessage(data.get('messages'))

        max_tokens = data.get('max_tokens', 512)
        temperature = data.get('temperature', 0.7)
        client = InferenceClient(api_key=env_data["HF_TOKEN"])
        chat_response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages
        )

        return jsonify(chat_response)
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
        
        messages = auxfunctions.filterMessage(data.get('messages'))
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
