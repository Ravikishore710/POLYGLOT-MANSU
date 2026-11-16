from huggingface_hub import HfApi

api = HfApi()

for repo in ["Ravikishore710/polyglot-mansu-tiny-aya-fire", "Ravikishore710/polyglot-mansu-fire-weights-private"]:
    print(f"\nFiles in {repo}:")
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="model")
        for f in files:
            print(f"  - {f}")
    except Exception as e:
        print(f"  Error: {e}")
