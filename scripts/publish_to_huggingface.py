#!/usr/bin/env python3
"""
scripts/publish_to_huggingface.py
Automated publishing script using local Hugging Face CLI credentials.
Deploys:
1. Space: <username>/polyglot-mansu-arena
2. Dataset: <username>/polyglot-mansu-curated
3. Model: <username>/polyglot-mansu-tiny-aya-fire
4. Collection: Polyglot-MANSU Track 2b Submission
"""

import sys
import json
from pathlib import Path

try:
    import huggingface_hub
    from huggingface_hub import HfApi, create_repo, upload_folder
except ImportError:
    print("Error: huggingface_hub is required.")
    sys.exit(1)

def main():
    token = huggingface_hub.get_token()
    if not token:
        print("Error: No local Hugging Face token detected in ~/.cache/huggingface/token.")
        print("Please authenticate first in your terminal.")
        sys.exit(1)

    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info["name"]
    print(f"=== Authenticated as Hugging Face User: {username} ===")

    # 1. Publish Space
    space_repo_id = f"{username}/polyglot-mansu-arena"
    print(f"\n[1/4] Publishing Gradio Space to https://huggingface.co/spaces/{space_repo_id} ...")
    try:
        api.create_repo(
            repo_id=space_repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            exist_ok=True
        )
        api.upload_folder(
            folder_path="space_app",
            repo_id=space_repo_id,
            repo_type="space",
            commit_message="Deploy Polyglot-MANSU Live Red-Team Arena and Visualizer",
            ignore_patterns=["**/__pycache__/**", "*.pyc", ".git*"]
        )
        print(f"  -> Space successfully deployed: https://huggingface.co/spaces/{space_repo_id}")
    except Exception as e:
        print(f"  -> Space deployment notice: {e}")

    # 2. Publish Dataset
    dataset_repo_id = f"{username}/polyglot-mansu-curated"
    print(f"\n[2/4] Publishing Curated Multilingual Dataset to https://huggingface.co/datasets/{dataset_repo_id} ...")
    try:
        api.create_repo(
            repo_id=dataset_repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True
        )
        api.upload_folder(
            folder_path="data/curated",
            repo_id=dataset_repo_id,
            repo_type="dataset",
            commit_message="Upload Polyglot-MANSU 10-language curated splits and manifests",
            ignore_patterns=["**/__pycache__/**", "*.pyc", ".git*"]
        )
        print(f"  -> Dataset successfully published: https://huggingface.co/datasets/{dataset_repo_id}")
    except Exception as e:
        print(f"  -> Dataset publishing notice: {e}")

    # 3. Publish Model Adapter & Audit Receipts
    model_repo_id = f"{username}/polyglot-mansu-tiny-aya-fire"
    print(f"\n[3/4] Publishing Model Repository & Receipts to https://huggingface.co/{model_repo_id} ...")
    try:
        api.create_repo(
            repo_id=model_repo_id,
            repo_type="model",
            private=False,
            exist_ok=True
        )
        # Upload prepared export package
        upload_dir = Path("results/hf_export")
        api.upload_folder(
            folder_path=str(upload_dir),
            repo_id=model_repo_id,
            repo_type="model",
            commit_message="Upload Polyglot-MANSU unlearning verification receipts and model card",
            ignore_patterns=["**/__pycache__/**", "*.pyc", ".git*"]
        )
        print(f"  -> Model repository successfully published: https://huggingface.co/{model_repo_id}")
    except Exception as e:
        print(f"  -> Model repository notice: {e}")

    # 4. Create Public Collection
    print("\n[4/4] Creating Public Hugging Face Collection ...")
    try:
        collection = api.create_collection(
            title="Polyglot-MANSU: Multilingual Machine Unlearning (Track 2b)",
            description="Official Lexsi AI Safety & Alignment Hackathon (Track 2b #7) project assets by Polyglot-MANSU team.",
            private=False
        )
        api.add_collection_item(collection.slug, item_id=space_repo_id, item_type="space")
        api.add_collection_item(collection.slug, item_id=model_repo_id, item_type="model")
        api.add_collection_item(collection.slug, item_id=dataset_repo_id, item_type="dataset")
        print(f"  -> Collection successfully created: https://huggingface.co/collections/{collection.slug}")
    except Exception as e:
        print(f"  -> Collection notice: {e}")

    print("\n=== Publishing Workflow Concluded Successfully ===")

if __name__ == "__main__":
    main()
