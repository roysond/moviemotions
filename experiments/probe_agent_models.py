"""
Which Bedrock models can this account actually run THIS agent on?

WHY THIS EXISTS, AND WHY IT WAS REWRITTEN
    list_foundation_models needs bedrock:ListFoundationModels, which this IAM user does
    not have — it can invoke, not enumerate. So do not ask what exists; try to use it.

    Version 1 of this probe asked each model to call ONE trivial tool with no arguments,
    and reported amazon.nova-lite-v1:0 as USABLE. The real agent then died on it with
    "Model produced invalid sequence as part of ToolUse". The probe was not wrong about
    the API; it was wrong about the QUESTION. "Can call a toy tool" and "can drive this
    agent" are different claims, and only the second one matters.

    So this version binds the REAL tools, sends the REAL system prompt, and asks a REAL
    question. A probe that does not reproduce production cannot clear a model for it.

    python experiments/probe_agent_models.py                 # built-in candidates
    python experiments/probe_agent_models.py <model-id> ...  # plus your own
"""

import os
import sys

import botocore
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import SYSTEM_PROMPT  # noqa: E402  — the real prompt, not a toy one
from tools import TOOLS          # noqa: E402  — all three, with full descriptions

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

# Two questions, because they exercise different routing. The first must reach
# search_films; the second must reach the graph. A model that answers either from its
# own knowledge instead of calling a tool is useless here however well it writes.
PROBES = [
    ("something tense where creatures hunt people", "search_films"),
    ("anything by Christopher Nolan?", "find_films_by_fact"),
]

print(f"region: {REGION}")
print(f"binding {len(TOOLS)} real tools · system prompt {len(SYSTEM_PROMPT)} chars\n")

usable = []
for model_id in CANDIDATES:
    try:
        llm = ChatBedrockConverse(model_id=model_id, region_name=REGION,
                                  temperature=0).bind_tools(TOOLS)
    except Exception as error:
        print(f"  {'--':8}  {model_id:<34} could not bind: {type(error).__name__}")
        continue

    verdicts, ok = [], True
    for question, wanted in PROBES:
        try:
            reply = llm.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(question)])
            calls = getattr(reply, "tool_calls", None) or []
            if not calls:
                verdicts.append(f"{wanted}: called NO tool")
                ok = False
            elif calls[0]["name"] != wanted:
                # Not fatal — routing is a judgement call, and the eval measures it
                # properly. Worth reporting so a surprise is visible here first.
                verdicts.append(f"chose {calls[0]['name']}, expected {wanted}")
            else:
                verdicts.append(f"{wanted} ok")
        except botocore.exceptions.ClientError as error:
            code = error.response["Error"]["Code"]
            verdicts.append({
                "AccessDeniedException": "not enabled for this account",
                "ValidationException": "id rejected — wrong name, or needs a us. profile",
                "ResourceNotFoundException": "no such model in this region",
            }.get(code, code))
            ok = False
            break
        except Exception as error:
            # This is where nova-lite failed: malformed tool-use on a real prompt.
            verdicts.append(f"{type(error).__name__}: {str(error)[:60]}")
            ok = False
            break

    if ok:
        usable.append(model_id)
    print(f"  {'USABLE' if ok else '--':8}  {model_id:<34} {' · '.join(verdicts)}")

print()
if usable:
    print("Swap the agent model with ONE of these, nothing else changed:")
    for model_id in usable:
        print(f"    BEDROCK_MODEL_AGENT={model_id} python eval_agent.py")
    print("\nUSABLE means it survived the real prompt with the real tools. It does NOT")
    print("mean it is better — measure that with eval_agent.py, against the baseline.")
else:
    print("Nothing usable. Enable model access in the AWS console:")
    print("    Bedrock -> Model access -> Modify model access")
