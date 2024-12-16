# Create a virtual environment
python -m venv vllm_env

# Activate the virtual environment
source vllm_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install vllm
pip install -r requirements.txt

# (Optional) Test installation
python -c "import vllm; print('vllm installed successfully')"

