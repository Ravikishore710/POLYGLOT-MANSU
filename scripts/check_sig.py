from huggingface_hub import HfApi
import inspect

api = HfApi()
print(inspect.signature(api.update_repo_settings))
