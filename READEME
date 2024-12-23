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
To create a virtual environment and update the README for your project in PyCharm, follow these steps:

2. **Create a Virtual Environment**:
   - In PyCharm, go to **File > Settings** (or **PyCharm > Preferences** on macOS).
   - Navigate to **Project > Python Interpreter**.
   - Click the gear icon ⚙️ and select **Add...**.
   - Choose **New Environment** and select the location and name for your virtual environment.
   - Click **OK** to create and set it as the project's interpreter.

3. **Install Required Packages**:
   - Open the **Terminal** in PyCharm or use the integrated Python Console.
   - Run:
     ```bash
     pip install flask vllm huggingface_hub python-dotenv
     ```
   - If you have additional dependencies, install them here.

4. **Generate a `requirements.txt` File**:
   - Run:
     ```bash
     pip freeze > requirements.txt
     ```
   - This file ensures others can recreate the environment with the same dependencies.


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
