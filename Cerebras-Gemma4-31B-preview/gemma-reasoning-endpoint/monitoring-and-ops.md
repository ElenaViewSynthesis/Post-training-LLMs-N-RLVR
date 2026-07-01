# Monitoring and Operations — Gemma-4-31B Reasoning Endpoint

## Performance Optimization

| Area | Recommendation |
|---|---|
| **Memory** | Model is already loaded in FP16 via TGI — 62GB sharded across 8× A100 40GB |
| **Batch size** | Start at 1, increase gradually; watch `ModelLatency` in CloudWatch before scaling up |
| **Context length** | `MAX_TOTAL_TOKENS=65536` is set — shorter prompts reduce KV cache pressure and cost |
| **Auto scaling** | Configure Application Auto Scaling on the endpoint for variable traffic (see below) |

---

## CloudWatch Monitoring

Track invocations, latency, and errors on the live endpoint:

```python
import boto3
from datetime import datetime, timedelta

cloudwatch = boto3.client("cloudwatch", region_name="us-east-1")

metrics = cloudwatch.get_metric_statistics(
    Namespace="AWS/SageMaker",
    MetricName="InvocationsPerInstance",
    Dimensions=[
        {
            "Name": "EndpointName",
            "Value": "gemma-4-31-b-reasoning",
        }
    ],
    StartTime=datetime.utcnow() - timedelta(hours=1),
    EndTime=datetime.utcnow(),
    Period=300,       # 5-minute buckets
    Statistics=["Average"],
)

for point in sorted(metrics["Datapoints"], key=lambda x: x["Timestamp"]):
    print(f"{point['Timestamp']}  avg invocations: {point['Average']:.2f}")
```

### Key metrics to watch

| Metric | What it tells you |
|---|---|
| `InvocationsPerInstance` | Request throughput per GPU instance |
| `ModelLatency` | Time the model takes to generate (microseconds) |
| `OverheadLatency` | SageMaker routing overhead |
| `Invocation4XXErrors` | Bad requests (payload format, missing params) |
| `Invocation5XXErrors` | Model errors (OOM, timeout, TGI crash) |

---

## Cost Optimization

| Strategy | Notes |
|---|---|
| **Spot instances** | Use for dev/testing — up to 70% cheaper, but can be interrupted |
| **Auto scaling** | Scale to 0 during off-hours; scale up on traffic via Application Auto Scaling |
| **Scheduled scaling** | Set scaling actions on a cron schedule if traffic is predictable |
| **Monitor usage** | Track `InvocationsPerInstance` — if it stays near 0, delete the endpoint |
| **Short sessions** | `ml.p4d.24xlarge` bills per second — delete immediately after testing |

---

## Cleanup

Delete the endpoint to stop all billing:

```python
import boto3

sm = boto3.client("sagemaker", region_name="us-east-1")
sm.delete_endpoint(EndpointName="gemma-4-31-b-reasoning")
print("Endpoint deleted. Billing stopped.")
```

Or via terminal:

```bash
python auto_terminate.py   # watchdog — deletes 2 min after InService
```

Or via AWS Console:
```
SageMaker Console → Endpoints → gemma-4-31-b-reasoning → Delete
```

Billing stops within minutes of deletion. The EBS volume and model artifacts in S3 (`s3://sagemaker-us-east-1-149901539173/gemma-4-31b/`) persist but incur only S3 storage costs (~$0.023/GB/month).

---

## Related Files

| File | Purpose |
|---|---|
| `deploy_sagemaker.py` | Deploys the endpoint |
| `auto_terminate.py` | Watchdog — auto-deletes after 2 min InService |
| `test_reasoning.py` | Runs reasoning, function calling, and multimodal tests |
| `check_quotas.py` | Checks ml.p4d.24xlarge quota before deploy |
