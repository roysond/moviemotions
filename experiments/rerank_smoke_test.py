"""Reachability test for the Cohere reranker — does it answer on this plan/region?"""

import os

import boto3
from dotenv import load_dotenv

load_dotenv()

REGION = os.environ["AWS_REGION"]
MODEL_ID = os.environ["BEDROCK_RERANK_MODEL"]
MODEL_ARN = f"arn:aws:bedrock:{REGION}::foundation-model/{MODEL_ID}"

QUERY = "a movie where creatures chase and hunt people, very intense"

DOCS = [
    "In the jungle, an elite military team is hunted one by one by a "
    "technologically advanced alien creature that stalks them as prey.",           # Predator
    "Three friends wake from a bachelor party with no memory, a baby in the "
    "closet and a tiger in the bathroom, and must find their missing friend.",     # The Hangover
    "A cowboy doll grows jealous when a spaceman toy arrives, but the two must "
    "cooperate when they are separated from their owner.",                         # Toy Story
]
LABELS = ["Predator", "The Hangover", "Toy Story"]

client = boto3.client("bedrock-agent-runtime", region_name=REGION)

try:
    response = client.rerank(
        queries=[{"type": "TEXT", "textQuery": {"text": QUERY}}],
        sources=[
            {
                "type": "INLINE",
                "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": doc}},
            }
            for doc in DOCS
        ],
        rerankingConfiguration={
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "modelConfiguration": {"modelArn": MODEL_ARN},
                "numberOfResults": len(DOCS),
            },
        },
    )
except Exception as error:
    print(f"FAILED: {type(error).__name__}")
    print(str(error)[:300])
else:
    print(f'QUERY: "{QUERY}"\n')
    for rank, result in enumerate(response["results"], start=1):
        print(f"{rank}. {result['relevanceScore']:.4f}  {LABELS[result['index']]}")