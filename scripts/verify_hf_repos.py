import os
from huggingface_hub import HfApi

api = HfApi()

print("--- Models ---")
for m in api.list_models(author="Ravikishore710"):
    print(f"Model: {m.id} | Private: {m.private}")

print("\n--- Datasets ---")
for d in api.list_datasets(author="Ravikishore710"):
    print(f"Dataset: {d.id} | Private: {d.private}")

print("\n--- Spaces ---")
for s in api.list_spaces(author="Ravikishore710"):
    print(f"Space: {s.id} | Private: {s.private}")
