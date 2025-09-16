#!/usr/bin/env python3
"""
Small utility to upload a local file to the Hugging Face Hub.

Usage example (PowerShell):
  $env:HUGGINGFACE_TOKEN="<your_token>"
  python scripts/upload_to_hf.py \
    --file "D:\\BitSkip\\quadratic_hbitlinear_final_model.pt" \
    --repo-id "<username_or_org>/<repo_name>" \
    --path-in-repo "quadratic_hbitlinear_final_model.pt" \
    --private

Notes:
- Token resolution order: --token arg > HUGGINGFACE_TOKEN env var
- If repo does not exist, this script can create it (unless --no-create is set)
"""

import os
import sys
import argparse
from typing import Optional

from huggingface_hub import HfApi, create_repo, hf_hub_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a file to Hugging Face Hub")
    parser.add_argument("--file", required=True, help="Local file path to upload")
    parser.add_argument("--repo-id", required=True, help="Target repo id, e.g. username/repo")
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Destination path in repo (defaults to same filename)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (overrides HUGGINGFACE_TOKEN env var)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch or tag to upload to (default: main)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private if it needs to be created",
    )
    parser.add_argument(
        "--no-create",
        action="store_true",
        help="Do not attempt to create the repo if it doesn't exist",
    )
    return parser.parse_args()


def resolve_token(cli_token: Optional[str]) -> Optional[str]:
    if cli_token:
        return cli_token
    return os.environ.get("HUGGINGFACE_TOKEN")


def main() -> None:
    args = parse_args()

    local_path = os.path.abspath(args.file)
    if not os.path.isfile(local_path):
        print(f"ERROR: File not found: {local_path}")
        sys.exit(1)

    token = resolve_token(args.token)
    if not token:
        print("ERROR: No token provided. Set HUGGINGFACE_TOKEN or pass --token.")
        sys.exit(1)

    dest_path = args.path_in_repo or os.path.basename(local_path)

    api = HfApi(token=token)

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


