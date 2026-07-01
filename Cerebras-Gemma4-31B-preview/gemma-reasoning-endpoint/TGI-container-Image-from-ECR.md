# TGI Container Image from ECR

## Overview

The SageMaker endpoint does not use a custom Docker image or a user-managed container registry. It pulls a pre-built TGI (Text Generation Inference) image from AWS's own public ECR (Elastic Container Registry).

## How the Image URI is Resolved

In `deploy_sagemaker.py`, the image is referenced as:

```python
image_uri=get_huggingface_llm_image_uri("huggingface", version="2.4.1")
```

This resolves at runtime to a URI of the form:

```
763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.4.1-tgi2.x-...
```

- **Account `763104351884`** — AWS's own Deep Learning Containers (DLC) account. This is not the user's account.
- **Region** — matches the deployment region (`us-east-1`).
- **Image** — `huggingface-pytorch-tgi-inference`, a managed TGI image maintained by AWS and HuggingFace jointly.

## What Happens at Deploy Time

1. SageMaker allocates an `ml.p4d.24xlarge` EC2 instance internally.
2. Pulls the TGI image from the AWS ECR URI above — no auth required from the user, the SageMaker execution role has ECR pull permissions via managed policies.
3. Downloads `google/gemma-4-31b-it` weights (62GB) from HuggingFace Hub into the attached EBS volume (`volume_size=256` GB). Requires `HF_TOKEN` for the gated model.
4. Starts the TGI server inside the container with the env vars defined in `deploy_sagemaker.py`.
5. SageMaker exposes the endpoint via HTTPS once the health check passes.

## What the User Does NOT Need

- No ECR repository to create or manage.
- No Docker image to build or push.
- No ECR credentials to configure.
- No EC2 instance to manage — SageMaker handles the underlying compute entirely.

## Required Credentials

| Credential | Purpose | Where configured |
|---|---|---|
| `HF_TOKEN` | Pull gated Gemma-4-31B weights from HuggingFace Hub | `Cerebras-Gemma4-31B-preview/.env` |
| IAM role `Gemma-4-31b-deploy` | Grants SageMaker permission to pull from ECR, write to S3, manage EC2 | Hardcoded in `deploy_sagemaker.py` |
| AWS access key (`aws configure`) | Authenticates the deploy script to call SageMaker APIs | Local `~/.aws/credentials` |

## Relevant Files

- `deploy_sagemaker.py` — calls `get_huggingface_llm_image_uri` and passes it to `HuggingFaceModel`
- `check_quotas.py` — verifies the `ml.p4d.24xlarge` quota before deploy
- `README.md` — end-to-end setup and deploy instructions
