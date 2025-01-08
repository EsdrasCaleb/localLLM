# VLLM Local Server

A Flask-based server for managing and running VLLM models with Hugging Face integration.

## Setup

### Prerequisites
- Python 3.8 or later
- Pip

### Steps
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up the `.env` file with your Hugging Face token:
   ```plaintext
   HF_TOKEN=your_huggingface_token
   ```

## Usage

### Start the Server
Run the following command:
```bash
python vllm_server_tool.py --host 0.0.0.0 --port 5000
```

### API Endpoints
- **List Models**: `GET /list_models` - Lists available text generation models from Hugging Face.
- **Download Model**: `POST /download_model` - Downloads the specified model.
- **Generate Text**: `POST /generate` - Generates text from a prompt using the specified model.

### Example Requests
#### Download Model
```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"model_name": "gpt2"}' http://localhost:5000/download_model
```

#### Generate Text
```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"model_name": "gpt2", "prompt": "Once upon a time"}' \
http://localhost:5000/generate
```

use 
```bash
sbatch --partition=gpu-4-a100 flaskbatchgpu.sh 
sbatch --partition=gpu-8-v100 flaskbatchgpu.sh 
sbatch --partition=gpu-8-h100 flaskbatchgpu.sh 
```
