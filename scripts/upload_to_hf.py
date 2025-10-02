#!/usr/bin/env python3
"""
Small utility to upload files to the Hugging Face Hub.
Supports both individual files and SafeTensors model directories.

Usage examples:

Single file upload:
  $env:HUGGINGFACE_TOKEN="<your_token>"
  python scripts/upload_to_hf.py \
    --file "model.pt" \
    --repo-id "username/repo" \
    --path-in-repo "model.pt"

SafeTensors model directory upload:
  python scripts/upload_to_hf.py \
    --model-dir "./safetensors_models/bitnet-1b" \
    --repo-id "username/bitnet-1b" \
    --private

Upload all SafeTensors models:
  python scripts/upload_to_hf.py \
    --upload-all-safetensors \
    --repo-prefix "username" \
    --private

Notes:
- Token resolution order: --token arg > HUGGINGFACE_TOKEN env var
- If repo does not exist, this script can create it (unless --no-create is set)
- SafeTensors upload includes config.json, README.md, and metadata files
"""

import os
import sys
import argparse
from typing import Optional, List
from pathlib import Path

from huggingface_hub import HfApi, create_repo, hf_hub_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload files to Hugging Face Hub")
    
    # File upload options
    parser.add_argument("--file", help="Local file path to upload")
    parser.add_argument("--model-dir", help="SafeTensors model directory to upload")
    parser.add_argument("--upload-all-safetensors", action="store_true", 
                       help="Upload all SafeTensors models from ./safetensors_models/")
    
    # Repo options
    parser.add_argument("--repo-id", help="Target repo id, e.g. username/repo")
    parser.add_argument("--repo-prefix", help="Repo prefix for batch uploads (e.g., 'username')")
    
    # Upload options
    parser.add_argument("--path-in-repo", help="Destination path in repo (defaults to same filename)")
    parser.add_argument("--token", help="Hugging Face token (overrides HUGGINGFACE_TOKEN env var)")
    parser.add_argument("--revision", default="main", help="Branch or tag to upload to (default: main)")
    parser.add_argument("--private", action="store_true", help="Create repo as private if it needs to be created")
    parser.add_argument("--no-create", action="store_true", help="Do not attempt to create the repo if it doesn't exist")
    
    return parser.parse_args()


def resolve_token(cli_token: Optional[str]) -> Optional[str]:
    if cli_token:
        return cli_token
    return os.environ.get("HUGGINGFACE_TOKEN")


def upload_safetensors_model(api: HfApi, model_dir: Path, repo_id: str, 
                           revision: str, private: bool, no_create: bool) -> None:
    """Upload a SafeTensors model directory to Hugging Face Hub."""
    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)
    
    # Check if repo exists
    try:
        api.repo_info(repo_id)
        repo_exists = True
    except Exception:
        repo_exists = False
    
    if not repo_exists:
        if no_create:
            print(f"ERROR: Repo {repo_id} does not exist and --no-create is set.")
            sys.exit(1)
        print(f"Repo {repo_id} not found. Creating (private={private})...")
        create_repo(repo_id=repo_id, token=api.token, private=private, exist_ok=True)
    
    # Files to upload from the model directory
    files_to_upload = [
        "*.safetensors",  # Main model file
        "config.json",    # Model config
        "README.md",      # Documentation
        "conversion_metadata.json"  # Conversion details
    ]
    
    uploaded_files = []
    
    for pattern in files_to_upload:
        if pattern.startswith("*"):
            # Handle wildcard patterns
            for file_path in model_dir.glob(pattern):
                if file_path.is_file():
                    print(f"Uploading {file_path.name} -> {repo_id}:{file_path.name} (rev: {revision}) ...")
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=file_path.name,
                        repo_id=repo_id,
                        repo_type="model",
                        revision=revision,
                        token=api.token,
                    )
                    uploaded_files.append(file_path.name)
        else:
            # Handle specific files
            file_path = model_dir / pattern
            if file_path.exists():
                print(f"Uploading {file_path.name} -> {repo_id}:{file_path.name} (rev: {revision}) ...")
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_path.name,
                    repo_id=repo_id,
                    repo_type="model",
                    revision=revision,
                    token=api.token,
                )
                uploaded_files.append(file_path.name)
    
    print(f"✅ Successfully uploaded {len(uploaded_files)} files to {repo_id}")
    for file_name in uploaded_files:
        url = hf_hub_url(repo_id=repo_id, filename=file_name, revision=revision)
        print(f"  📁 {file_name}: {url}")


def find_safetensors_models(safetensors_dir: Path) -> List[Path]:
    """Find all SafeTensors model directories."""
    models = []
    
    if not safetensors_dir.exists():
        print(f"ERROR: SafeTensors directory not found: {safetensors_dir}")
        return models
    
    for model_dir in safetensors_dir.iterdir():
        if model_dir.is_dir():
            # Check if it contains a .safetensors file
            safetensors_files = list(model_dir.glob("*.safetensors"))
            if safetensors_files:
                models.append(model_dir)
                print(f"Found SafeTensors model: {model_dir.name}")
    
    return models


def upload_all_safetensors_models(api: HfApi, repo_prefix: str, revision: str, 
                                private: bool, no_create: bool) -> None:
    """Upload all SafeTensors models to separate repos."""
    safetensors_dir = Path("./safetensors_models")
    models = find_safetensors_models(safetensors_dir)
    
    if not models:
        print("No SafeTensors models found in ./safetensors_models/")
        return
    
    print(f"Found {len(models)} SafeTensors models to upload")
    
    for model_dir in models:
        model_name = model_dir.name
        repo_id = f"{repo_prefix}/{model_name}"
        
        print(f"\n📦 Uploading {model_name} to {repo_id}")
        try:
            upload_safetensors_model(api, model_dir, repo_id, revision, private, no_create)
        except Exception as e:
            print(f"❌ Failed to upload {model_name}: {e}")
    
    print(f"\n✅ Completed uploading {len(models)} SafeTensors models")


def main() -> None:
    args = parse_args()
    
    # Validate arguments
    if not any([args.file, args.model_dir, args.upload_all_safetensors]):
        print("ERROR: Must specify one of --file, --model-dir, or --upload-all-safetensors")
        sys.exit(1)
    
    if args.upload_all_safetensors and not args.repo_prefix:
        print("ERROR: --repo-prefix is required when using --upload-all-safetensors")
        sys.exit(1)
    
    if (args.file or args.model_dir) and not args.repo_id:
        print("ERROR: --repo-id is required when uploading single file or model directory")
        sys.exit(1)
    
    # Get token
    token = resolve_token(args.token)
    if not token:
        print("ERROR: No token provided. Set HUGGINGFACE_TOKEN or pass --token.")
        sys.exit(1)
    
    api = HfApi(token=token)
    
    # Handle different upload modes
    if args.upload_all_safetensors:
        # Upload all SafeTensors models
        upload_all_safetensors_models(api, args.repo_prefix, args.revision, args.private, args.no_create)
        
    elif args.model_dir:
        # Upload single SafeTensors model directory
        model_dir = Path(args.model_dir)
        upload_safetensors_model(api, model_dir, args.repo_id, args.revision, args.private, args.no_create)
        
    elif args.file:
        # Upload single file (original functionality)
        local_path = os.path.abspath(args.file)
        if not os.path.isfile(local_path):
            print(f"ERROR: File not found: {local_path}")
            sys.exit(1)
        
        dest_path = args.path_in_repo or os.path.basename(local_path)
        
        # Create repo if needed
        try:
            api.repo_info(args.repo_id)
            repo_exists = True
        except Exception:
            repo_exists = False
        
        if not repo_exists:
            if args.no_create:
                print(f"ERROR: Repo {args.repo_id} does not exist and --no-create is set.")
                sys.exit(1)
            print(f"Repo {args.repo_id} not found. Creating (private={args.private})...")
            create_repo(repo_id=args.repo_id, token=token, private=bool(args.private), exist_ok=True)
        
        print(f"Uploading {local_path} -> {args.repo_id}:{dest_path} (rev: {args.revision}) ...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=dest_path,
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.revision,
            token=token,
        )
        
        url = hf_hub_url(repo_id=args.repo_id, filename=dest_path, revision=args.revision)
        print("Done. File available at:")
        print(url)


if __name__ == "__main__":
    main()


