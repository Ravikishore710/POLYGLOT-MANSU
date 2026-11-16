import re

readme_path = r"C:\Users\RahulNew\.cache\huggingface\hub\models--Ravikishore710--polyglot-mansu-tiny-aya-fire\snapshots"
import glob
files = glob.glob(readme_path + "/**/README.md", recursive=True)
if files:
    with open(files[0], "r", encoding="utf-8") as f:
        text = f.read()
    
    print("Mentions of weights / checkpoint:")
    for line in text.split("\n"):
        if any(w in line.lower() for w in ["checkpoint", "weights", "safetensors", "stage b", "a100"]):
            print("  ", line[:120])
