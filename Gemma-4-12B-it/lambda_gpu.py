"""
Lambda Cloud — spin up a GPU instance and SSH into it.

Usage:
    uv run lambda_gpu.py                        # launch default instance type
    uv run lambda_gpu.py --type gpu_1x_h100_sxm4
    uv run lambda_gpu.py --list-types           # show available GPU types + prices
    uv run lambda_gpu.py --list                 # show your running instances
    uv run lambda_gpu.py --terminate <id>       # terminate an instance
"""

import argparse
import os
import subprocess
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LAMBDA_API_KEY")
SSH_KEY_NAME = os.getenv("LAMBDA_SSH_KEY_NAME")
DEFAULT_INSTANCE_TYPE = os.getenv("LAMBDA_INSTANCE_TYPE", "gpu_1x_a100_sxm4")
SSH_KEY_PATH = os.path.expanduser(os.getenv("LAMBDA_SSH_KEY_PATH", "~/.ssh/id_rsa"))

BASE = "https://cloud.lambda.ai/api/v1"


def headers():
    return {"Authorization": f"Bearer {API_KEY}"}


def check_env():
    missing = [k for k, v in {"LAMBDA_API_KEY": API_KEY, "LAMBDA_SSH_KEY_NAME": SSH_KEY_NAME}.items() if not v or v.startswith("your_")]
    if missing:
        print(f"[error] Missing or placeholder values in .env: {', '.join(missing)}")
        sys.exit(1)


# ── API helpers ───────────────────────────────────────────────────────────────

def list_instances():
    r = requests.get(f"{BASE}/instances", headers=headers())
    r.raise_for_status()
    return r.json()["data"]


def list_instance_types():
    r = requests.get(f"{BASE}/instance-types", headers=headers())
    r.raise_for_status()
    return r.json()["data"]


def list_ssh_keys():
    r = requests.get(f"{BASE}/ssh-keys", headers=headers())
    r.raise_for_status()
    return r.json()["data"]


def region_names(regions) -> list[str]:
    if isinstance(regions, dict):
        return list(regions.keys())
    if isinstance(regions, list):
        names = []
        for region in regions:
            if isinstance(region, str):
                names.append(region)
            elif isinstance(region, dict):
                name = region.get("name") or region.get("region_name")
                if name:
                    names.append(name)
        return names
    return []


def launch(instance_type: str, region: str) -> str:
    payload = {
        "region_name": region,
        "instance_type_name": instance_type,
        "ssh_key_names": [SSH_KEY_NAME],
        "quantity": 1,
    }
    r = requests.post(f"{BASE}/instance-operations/launch", headers=headers(), json=payload)
    if r.status_code != 200:
        print(f"[error] Launch failed: {r.status_code} {r.text}")
        sys.exit(1)
    ids = r.json()["data"]["instance_ids"]
    return ids[0]


def get_instance(instance_id: str) -> dict:
    r = requests.get(f"{BASE}/instances/{instance_id}", headers=headers())
    r.raise_for_status()
    return r.json()["data"]


def terminate(instance_id: str):
    r = requests.post(f"{BASE}/instance-operations/terminate", headers=headers(), json={"instance_ids": [instance_id]})
    r.raise_for_status()
    print(f"Terminated {instance_id}")


def wait_until_running(instance_id: str, timeout: int = 300) -> dict:
    print(f"Waiting for instance {instance_id} to become active", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        inst = get_instance(instance_id)
        status = inst.get("status")
        if status == "active":
            print(" ready.")
            return inst
        if status == "terminated":
            print("\n[error] Instance terminated unexpectedly.")
            sys.exit(1)
        print(".", end="", flush=True)
        time.sleep(10)
    print(f"\n[error] Timed out after {timeout}s waiting for instance.")
    sys.exit(1)


def ssh_into(ip: str):
    cmd = [
        "ssh",
        "-i", SSH_KEY_PATH,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=60",
        f"ubuntu@{ip}",
    ]
    print(f"Connecting: {' '.join(cmd)}\n")
    subprocess.run(cmd)


# ── Subcommands ───────────────────────────────────────────────────────────────

def cmd_list(args):
    instances = list_instances()
    if not instances:
        print("No running instances.")
        return
    print(f"{'ID':<25} {'Type':<25} {'Status':<12} {'IP'}")
    print("-" * 80)
    for i in instances:
        print(f"{i['id']:<25} {i['instance_type']['name']:<25} {i['status']:<12} {i.get('ip', 'N/A')}")


def cmd_list_types(args):
    types = list_instance_types()
    print(f"{'Name':<30} {'vCPUs':>6} {'RAM GB':>8} {'$/hr':>8}  Regions")
    print("-" * 80)
    for name, info in sorted(types.items()):
        spec = info["instance_type"]
        regions = ", ".join(region_names(info.get("regions_with_capacity_available"))) or "none available"
        price = spec["price_cents_per_hour"] / 100
        print(f"{name:<30} {spec['specs']['vcpus']:>6} {spec['specs']['memory_gib']:>8} {price:>8.2f}  {regions}")


def cmd_launch(args):
    check_env()
    instance_type = args.type

    # find a region with capacity
    types = list_instance_types()
    if instance_type not in types:
        print(f"[error] Unknown instance type '{instance_type}'. Run --list-types to see options.")
        sys.exit(1)
    regions = region_names(types[instance_type].get("regions_with_capacity_available"))
    if not regions:
        print(f"[error] No capacity available for {instance_type} right now. Try --list-types for alternatives.")
        sys.exit(1)
    region = regions[0]

    print(f"Launching {instance_type} in {region} ...")
    instance_id = launch(instance_type, region)
    print(f"Instance ID: {instance_id}")

    inst = wait_until_running(instance_id)
    ip = inst.get("ip")
    print(f"IP: {ip}")

    if not args.no_ssh:
        ssh_into(ip)


def cmd_terminate(args):
    check_env()
    terminate(args.id)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Lambda Cloud GPU manager")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List running instances")
    sub.add_parser("list-types", help="List available GPU types and prices")

    p_launch = sub.add_parser("launch", help="Launch a GPU instance and SSH in")
    p_launch.add_argument("--type", default=DEFAULT_INSTANCE_TYPE, help="Instance type name")
    p_launch.add_argument("--no-ssh", action="store_true", help="Don't SSH after launch")

    p_term = sub.add_parser("terminate", help="Terminate an instance")
    p_term.add_argument("id", help="Instance ID")

    args = parser.parse_args()

    if args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "list-types":
        cmd_list_types(args)
    elif args.cmd == "terminate":
        cmd_terminate(args)
    else:
        # default: launch
        args.type = getattr(args, "type", DEFAULT_INSTANCE_TYPE)
        args.no_ssh = getattr(args, "no_ssh", False)
        cmd_launch(args)


if __name__ == "__main__":
    main()
