from huggingface_hub import HfApi
import inspect

api = HfApi()
print(inspect.signature(api.edit_discussion_comment))
