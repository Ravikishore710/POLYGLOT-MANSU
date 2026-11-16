from huggingface_hub import HfApi
import inspect

api = HfApi()
methods = [m for m in dir(api) if not m.startswith('_')]
print("Methods matching update/repo/private:")
for m in methods:
    if any(k in m for k in ['repo', 'private', 'visibility', 'update']):
        print(" ", m)
