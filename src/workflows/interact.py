"""Send signals to or query a running workflow execution."""
# ruff: noqa: E402

import argparse
import asyncio
import json
import os

from dotenv import load_dotenv

load_dotenv(override=True)

from mistralai.workflows.client import get_mistral_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interact with a running workflow.")
    parser.add_argument("--execution-id", required=True, help="Execution ID to target")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--signal", metavar="NAME", help="Signal name to send (e.g. reviewer-decision)")
    group.add_argument("--query", metavar="NAME", help="Query name to call (e.g. review-status)")
    parser.add_argument("--input", default=r"{}", help='JSON payload (e.g. \'{"approved": true}\')')
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    try:
        payload = json.loads(args.input)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Error: invalid JSON for --input: {exc}") from exc

    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        raise SystemExit("Error: MISTRAL_API_KEY is not set.")

    client = get_mistral_client(
        api_key=api_key,
        server_url=os.environ.get("SERVER_URL", "https://api.mistral.ai"),
    )

    if args.signal:
        result = await client.workflows.executions.signal_workflow_execution_async(
            execution_id=args.execution_id,
            name=args.signal,
            input=payload,
        )
        print(f"Signal sent: {result}")
    else:
        result = await client.workflows.executions.query_workflow_execution_async(
            execution_id=args.execution_id,
            name=args.query,
            input=payload,
        )
        print(f"Query result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
