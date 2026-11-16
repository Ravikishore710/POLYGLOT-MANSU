from huggingface_hub import hf_hub_download

readme_path = hf_hub_download(repo_id="Ravikishore710/polyglot-mansu-tiny-aya-fire", filename="README.md", repo_type="model", force_download=True)
with open(readme_path, "r", encoding="utf-8") as f:
    text = f.read()

print("Contains 'Scientific Transparency':", "Scientific Transparency" in text)
print("Contains 'polyglot-mansu-fire-weights-private':", "polyglot-mansu-fire-weights-private" in text)
print("Contains 'Total Financial Cost':", "Total Financial Cost" in text)
print("Contains 'Member 1: Rahul Ratan':", "Member 1: Rahul Ratan" in text)
print("Contains 'Member 2: Ravi Kishore':", "Member 2: Ravi Kishore" in text)
print("Contains outdated 'due to Kaggle session storage limits':", "due to Kaggle session storage limits" in text)
