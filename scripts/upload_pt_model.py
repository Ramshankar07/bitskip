#!/usr/bin/env python3
"""
Upload PyTorch .pt model to Hugging Face Hub.
This is a simpler approach that uploads the original checkpoint as-is.
"""

import os
import sys
import argparse
from huggingface_hub import HfApi, create_repo, hf_hub_url


def parse_args():
    parser = argparse.ArgumentParser(description="Upload PyTorch model to Hugging Face Hub")
    parser.add_argument("--file", required=True, help="Local .pt file path to upload")
    parser.add_argument("--repo-id", required=True, help="Target repo id, e.g. username/repo")
    parser.add_argument("--path-in-repo", default=None, help="Destination path in repo")
    parser.add_argument("--token", default=None, help="Hugging Face token")
    parser.add_argument("--revision", default="main", help="Branch or tag to upload to")
    parser.add_argument("--private", action="store_true", help="Create repo as private")
    parser.add_argument("--no-create", action="store_true", help="Don't create repo if it doesn't exist")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Get token
    token = args.token or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("ERROR: No token provided. Set HUGGINGFACE_TOKEN or pass --token.")
        sys.exit(1)
    
    # Check file exists
    file_path = os.path.abspath(args.file)
    if not os.path.isfile(file_path):
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)
    
    # Determine destination path
    dest_path = args.path_in_repo or os.path.basename(file_path)
    
    # Initialize API
    api = HfApi(token=token)
    
    # Check if repo exists
    try:
        api.repo_info(args.repo_id)
        repo_exists = True
    except Exception:
        repo_exists = False
    
    # Create repo if needed
    if not repo_exists:
        if args.no_create:
            print(f"ERROR: Repo {args.repo_id} does not exist and --no-create is set.")
            sys.exit(1)
        print(f"Creating repo {args.repo_id} (private={args.private})...")
        create_repo(repo_id=args.repo_id, token=token, private=args.private, exist_ok=True)
    
    # Upload file
    print(f"Uploading {file_path} -> {args.repo_id}:{dest_path}...")
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=dest_path,
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        token=token,
    )
    
    # Show URL
    url = hf_hub_url(repo_id=args.repo_id, filename=dest_path, revision=args.revision)
    print("✅ Upload successful!")
    print(f"Model available at: {url}")
    
    # Show usage instructions
    print("\n📖 Usage instructions:")
    print(f"```python")
    print(f"from huggingface_hub import hf_hub_download")
    print(f"import torch")
    print(f"")
    print(f"# Download the model")
    print(f"model_path = hf_hub_download(")
    print(f"    repo_id='{args.repo_id}',")
    print(f"    filename='{dest_path}',")
    print(f"    local_dir='./models'")
    print(f")")
    print(f"")
    print(f"# Load the model")
    print(f"checkpoint = torch.load(model_path, map_location='cpu')")
    print(f"```")


if __name__ == "__main__":
    main()
