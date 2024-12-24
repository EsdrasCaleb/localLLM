import requests
from main import download_model
file_repo={
    #"Phi-3.5-mini-instruct-Q8_0.gguf":"bartowski/Phi-3.5-mini-instruct-GGUF",
    #"OpenCoder-8B-Instruct-Q6_K.gguf":"lmstudio-community/OpenCoder-8B-Instruct-GGUF",
    #"Yi-Coder-9B-Chat-Q4_K_M.gguf":"lmstudio-community/Yi-Coder-9B-Chat-GGUF",
    #"EXAONE-3.5-2.4B-Instruct-BF16.ggf":"LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF",
    #"granite-3.1-8b-instruct-Q6_K.gguf":"lmstudio-community/granite-3.1-8b-instruct-GGUF",
    #"Llama-3.2-3B-Instruct-f16.gguf":"second-state/Llama-3.2-3B-Instruct-GGUF",
    #"gemma-2-9b-it-Q4_K_M-fp16.gguf":"bartowski/gemma-2-9b-it-GGUF",
    #"Ministral-8B-Instruct-2410-Q6_K_L.gguf":"bartowski/Ministral-8B-Instruct-2410-GGUF",
    #"codegemma-7b-it-Q6_K.gguf":"second-state/CodeGemma-7b-it-GGUF",
    #"matteogeniaccio.phi-4.Q3_K_M.gguf":"matteogeniaccio/phi-4",
    #"internlm2_5-7b-chat-q8_0.gguf":"internlm/internlm2_5-7b-chat-gguf",
    "meta-llama/Llama-3.2-1B-Instruct":None
}

for item,module in file_repo.items():
    download_model(module,item)