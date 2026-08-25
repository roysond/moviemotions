"""
Which Bedrock models can this account actually run the AGENT on?

list_foundation_models needs bedrock:ListFoundationModels, which this IAM user does not
have — it can invoke, not enumerate. So do not ask what exists; try to use it and report.
A rejected call costs nothing and tells the truth about THIS account, in THIS region.

The probe is not "does the model reply". It is "does the model CALL A TOOL", because an
agent that cannot call a tool is useless here no matter how well it writes.

    python experiments/probe_agent_models.py                 # the built-in candidates
    python experiments/probe_agent_models.py <model-id> ...  # plus your own
"""

import os
import sys

import boto3
import botocore
from dotenv import load_dotenv

load_dotenv()
REGION = os.environ["AWS_REGION"]

# A plain id may be rejected in favour of a REGIONAL INFERENCE PROFILE — the same model
# reached through a "us." prefix. Both forms are probed; whichever answers is the one to use.
CANDIDATES = [
    "amazon.nova-micro-v1:0",            # the current agent — the control
    "amazon.nova-lite-v1:0",
    "amazon.nova-pro-v1:0",
    "amazon.nova-premier-v1:0",
    "us.amazon.nova-lite-v1:0",
    "us.amazon.nova-pro-v1:0",
    "us.amazon.nova-premier-v1:0",
] + sys.argv[1:]

# The smallest possible tool. If a model will call this, it will call search_films.
TOOL_CONFIG = {"tools": [{"toolSpec": {
    "name": "ping",
    "description": "Returns the word pong. Call this tool to answer.",
    "inputSchema": {"json": {"type": "object", "properties": {}, "required": []}},
}}]}

client = boto3.client("bedrock-runtime", region_name=REGION)

print(f"region: {REGION}\n")
usable = []
for model_id in CANDIDATES:
    try:
        reply = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "Call the ping tool."}]}],
            toolConfig=TOOL_CONFIG,
            inferenceConfig={"maxTokens": 256},
        )
        blocks = reply["output"]["message"]["content"]
        called = any("toolUse" in block for block in blocks)
        if called:
            usable.append(model_id)
        print(f"  {'USABLE' if called else 'replied':8}  {model_id:<34} "
              f"{'called the tool' if called else 'answered but called NO tool'}")
    except botocore.exceptions.ClientError as error:
        code = error.response["Error"]["Code"]
        note = {"AccessDeniedException": "not enabled for this account",
                "ValidationException": "id rejected — wrong name, or needs a us. profile",
                "ResourceNotFoundException": "no such model in this region"}.get(code, code)
        print(f"  {'--':8}  {model_id:<34} {note}")

print()
if usable:
    print("Swap the agent model with ONE of these, nothing else changed:")
    for model_id in usable:
        print(f"    BEDROCK_MODEL_AGENT={model_id} python eval_agent.py")
else:
    print("Nothing usable. Enable model access in the AWS console:")
    print("    Bedrock -> Model access -> Modify model access")
