#!/usr/bin/env python3
"""
scripts/verify_hf_auth.py
Checks local Hugging Face CLI authentication and token permissions.
"""

import sys
try:
    import huggingface_hub
except ImportError:
    print("Error: huggingface_hub is not installed in the environment.")
    sys.exit(1)

def main():
    token = huggingface_hub.get_token()
    if not token:
        print("[AUTH STATUS] FAILED: No local Hugging Face token found.")
        sys.exit(1)

    try:
        user_info = huggingface_hub.whoami(token=token)
        username = user_info.get("name", "Unknown")
        fullname = user_info.get("fullname", "")
        auth_info = user_info.get("auth", {})
        auth_type = auth_info.get("type", "unknown")
        access_token_info = auth_info.get("accessToken", {})
        role = access_token_info.get("role", "oauth" if auth_type == "oauth" else "unknown")

        print(f"[AUTH STATUS] SUCCESS: Authenticated as '{username}' ({fullname})")
        print(f"[AUTH STATUS] Auth Mechanism: '{auth_type}' (Role: '{role}')")

        if auth_type == "oauth" or role == "write":
            print("[AUTH STATUS] Confirmed: Valid WRITE permissions active. Ready for publishing.")
            sys.exit(0)
        else:
            print("[AUTH STATUS] Notice: Non-OAuth token without explicit write role detected.")
            sys.exit(0)
    except Exception as e:
        print(f"[AUTH STATUS] ERROR: Token validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
