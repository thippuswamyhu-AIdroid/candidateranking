from time import time
from os import makedirs
from subprocess import Popen

from sentence_transformers import SentenceTransformer

def download_model_files():
    
    start = time()
    
    save_dir = "./models/BGE-embed/BAAI/bge-small-en-v1.5/"
    embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    embedding_model.save(save_dir)

    print(f"\nSaved BGE-Small-Embed-English-v1.5 Embedding file to: {save_dir} in {(time() - start):.2f} secs.\n")
    start = time()
    
    makedirs("models/Qwen-GGUF/", exist_ok=True)
    command = [
        "curl", "-L",
        "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf",
        "-o", "models/Qwen-GGUF/Qwen3-0.6B-Q4_K_M.gguf"
    ]
    
    process = Popen(command)
    return_code = process.wait()
    if return_code == 0:
        print(f"\nSaved Qwen3-0.6B-Q4-GGUF Model file to: {save_dir} in {(time() - start):.2f} secs.\n")
    else:
        print(f"Failed to download Qwen3-0.6B-Q4-GGUF Model file with exit code {return_code}.\n")
    
if __name__ == "__main__":
    
    download_model_files()