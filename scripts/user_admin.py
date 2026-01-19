import argparse
import json
import os
from datetime import datetime
from getpass import getpass

from werkzeug.security import generate_password_hash


def _default_users_file() -> str:
    return os.getenv('FACEPASS_USERS_FILE', os.path.join('data', 'users.json'))


def _load_users(path: str) -> dict:
    if not os.path.exists(path):
        return {"users": {}}

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return {"users": {}}

    users = data.get("users")
    if not isinstance(users, dict):
        users = {}

    return {"users": users}


def _atomic_write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    tmp_path = path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    os.replace(tmp_path, path)


def cmd_create(args: argparse.Namespace) -> int:
    username = (args.username or "").strip()
    if not username:
        raise SystemExit("--username is required")

    password = args.password
    if password is None:
        pw1 = getpass("Password: ")
        pw2 = getpass("Confirm password: ")
        if pw1 != pw2:
            raise SystemExit("Passwords do not match")
        password = pw1

    if not password:
        raise SystemExit("Password cannot be empty")

    path = args.file
    data = _load_users(path)
    users = data["users"]

    if username in users and not args.force:
        raise SystemExit(f"User '{username}' already exists. Use --force to overwrite.")

    role = args.role
    now = datetime.utcnow().isoformat() + "Z"

    users[username] = {
        "password_hash": generate_password_hash(password),
        "role": role,
        "created_at": users.get(username, {}).get("created_at", now),
        "updated_at": now,
    }

    _atomic_write_json(path, data)
    print(f"✓ Created user '{username}' (role={role}) in {path}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    path = args.file
    data = _load_users(path)
    users = data.get("users", {})

    if not users:
        print(f"No users found in {path}")
        return 0

    for username in sorted(users.keys()):
        info = users.get(username) or {}
        role = info.get("role", "user")
        created_at = info.get("created_at", "")
        print(f"- {username} (role={role}) {created_at}")

    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    username = (args.username or "").strip()
    if not username:
        raise SystemExit("--username is required")

    path = args.file
    data = _load_users(path)
    users = data.get("users", {})

    if username not in users:
        raise SystemExit(f"User '{username}' not found in {path}")

    del users[username]
    _atomic_write_json(path, data)
    print(f"✓ Deleted user '{username}' from {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="FacePass user admin (creates local users.json with hashed passwords)")
    parser.set_defaults(func=None)

    parser.add_argument(
        "--file",
        default=_default_users_file(),
        help="Users file path (default: FACEPASS_USERS_FILE env var or data/users.json)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Create a user")
    p_create.add_argument("--username", required=True)
    p_create.add_argument("--password", default=None, help="If omitted, you will be prompted")
    p_create.add_argument("--role", default="user", choices=["user", "admin"])
    p_create.add_argument("--force", action="store_true", help="Overwrite if user exists")
    p_create.set_defaults(func=cmd_create)

    p_list = sub.add_parser("list", help="List users")
    p_list.set_defaults(func=cmd_list)

    p_delete = sub.add_parser("delete", help="Delete a user")
    p_delete.add_argument("--username", required=True)
    p_delete.set_defaults(func=cmd_delete)

    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
