#!/usr/bin/env python3
"""
Polls SageMaker endpoint status every 60 seconds.
When InService: waits exactly 2 minutes, then deletes the endpoint.
Run this before sleeping: python auto_terminate.py
"""
import os
import time
import boto3
from pathlib import Path
from botocore.exceptions import ClientError

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "gemma-4-31-b-reasoning")
REGION        = os.getenv("AWS_REGION", "us-east-1")
LIVE_SECONDS  = 120  # 2 minutes
POLL_SECONDS  = 60   # check every 60 s while waiting for deploy

sm = boto3.client("sagemaker", region_name=REGION)


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


log(f"Watchdog active — endpoint: {ENDPOINT_NAME}  region: {REGION}")
log(f"Will allow {LIVE_SECONDS // 60} min of live traffic, then auto-terminate.")
log("Ctrl+C to abort.\n")

while True:
    try:
        resp   = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = resp["EndpointStatus"]
        log(f"Status: {status}")

        if status == "InService":
            log(f"Endpoint is live. Waiting {LIVE_SECONDS} seconds before deleting...")
            time.sleep(LIVE_SECONDS)
            sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
            log("Endpoint deleted. Billing stopped. Exiting.")
            break

        elif status in ("Failed", "OutOfService", "Deleting"):
            log(f"Endpoint is {status} — nothing to terminate. Exiting.")
            break

        # Still creating — keep polling
    except ClientError as e:
        if "Could not find endpoint" in str(e):
            log("Endpoint not yet created — waiting for deploy to finish...")
        else:
            log(f"AWS error: {e}")

    time.sleep(POLL_SECONDS)
