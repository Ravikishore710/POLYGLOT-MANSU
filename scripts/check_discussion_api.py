from huggingface_hub import HfApi

api = HfApi()
methods = [m for m in dir(api) if 'discussion' in m.lower() or 'comment' in m.lower()]
print("Discussion methods:", methods)
