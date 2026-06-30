"""Check SageMaker endpoint quotas for g6 instance types in us-east-1."""
import boto3

REGION = "us-east-1"

# Quota codes for SageMaker endpoint usage
QUOTAS = {
    "ml.p5.48xlarge": "L-BC4DA661",   # 8x H100 80GB — production target
    "ml.g6.12xlarge": "L-43FA1649",
    "ml.g6.24xlarge": "L-1194F27D",
    "ml.g6.48xlarge": "L-4C3A2D2B",
}

client = boto3.client("service-quotas", region_name=REGION)

print(f"SageMaker endpoint quotas in {REGION}:\n")
best = None
for instance, code in QUOTAS.items():
    try:
        resp  = client.get_service_quota(ServiceCode="sagemaker", QuotaCode=code)
        quota = int(resp["Quota"]["Value"])
        status = "READY" if quota >= 1 else "need increase"
        print(f"  {instance}: {quota}  ({status})")
        if quota >= 1 and best is None:
            best = instance
    except Exception as e:
        print(f"  {instance}: error — {e}")

print()
if best:
    print(f"Use instance_type='{best}' in deploy_sagemaker.py")
else:
    print("No quota available. Request an increase:")
    print("https://console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas")
