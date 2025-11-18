import argparse
import os
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, TransformersModel
from custom_tools import (
    dataset_download_tool,
    check_hf_dataset_subset,
    reformat_dataset_tool,
    validate_dataset_tool,
    save_dataset_tool,
    detect_problem_type_tool,
    map_row_tool,
    load_dataset_tool,
)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


load_dotenv()

parser = argparse.ArgumentParser(description="Run the BenchHub reformatting agent.")
parser.add_argument(
    "--backend",
    choices=["lite-llm", "transformers"],
    default=os.getenv("AGENT_BACKEND", "lite-llm"),
    help="Model backend to use. Set to 'transformers' for local HF models.",
)
parser.add_argument(
    "--model-id",
    default=os.getenv("LLM_MODEL_ID", "gpt-4o"),
    help="Identifier for the selected model/backend.",
)
parser.add_argument(
    "--llm-provider",
    default=os.getenv("LLM_PROVIDER"),
    help="LiteLLM provider (only if backend=lite-llm).",
)
parser.add_argument(
    "--api-base",
    default=os.getenv("LLM_API_BASE"),
    help="Base URL for LiteLLM-compatible endpoints.",
)
parser.add_argument(
    "--api-key",
    default=os.getenv("LLM_API_KEY"),
    help="API key for LiteLLM backend (falls back to OPENAI/HUGGINGFACE API keys).",
)
parser.add_argument(
    "--device-map",
    default=os.getenv("TRANSFORMERS_DEVICE_MAP"),
    help="Transformers device_map (e.g., 'auto', 'cuda', 'cpu').",
)
parser.add_argument(
    "--torch-dtype",
    default=os.getenv("TRANSFORMERS_TORCH_DTYPE"),
    help="Transformers torch_dtype (e.g., 'bfloat16').",
)
parser.add_argument(
    "--trust-remote-code",
    action=argparse.BooleanOptionalAction,
    default=env_flag("TRANSFORMERS_TRUST_REMOTE_CODE", False),
    help="Allow transformers to execute remote code.",
)
parser.add_argument(
    "--max-new-tokens",
    type=int,
    default=int(os.getenv("TRANSFORMERS_MAX_NEW_TOKENS", "4096")),
    help="Maximum new tokens for Transformers backend.",
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=int(os.getenv("AGENT_MAX_STEPS", "30")),
    help="Maximum reasoning steps for the agent.",
)
args = parser.parse_args()

default_api_key = os.getenv("OPENAI_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")


def build_model():
    if args.backend == "transformers":
        transformer_kwargs = {
            "model_id": args.model_id,
            "max_new_tokens": args.max_new_tokens,
            "trust_remote_code": args.trust_remote_code,
        }
        if args.device_map:
            transformer_kwargs["device_map"] = args.device_map
        if args.torch_dtype:
            transformer_kwargs["torch_dtype"] = args.torch_dtype
        return TransformersModel(**transformer_kwargs)

    llm_provider = args.llm_provider
    api_base = args.api_base
    api_key = args.api_key or default_api_key

    if args.model_id.startswith("huggingface/") and not llm_provider:
        llm_provider = "huggingface"

    provider_hint = (llm_provider or "").lower()
    if not api_key and (args.model_id.startswith("huggingface/") or provider_hint == "huggingface"):
        api_key = huggingface_api_key

    model_kwargs = {"model_id": args.model_id}
    if llm_provider:
        model_kwargs["llm_provider"] = llm_provider
    if api_base:
        model_kwargs["api_base"] = api_base
    if api_key:
        model_kwargs["api_key"] = api_key
    return LiteLLMModel(**model_kwargs)


# Build Agent
model = build_model()

agent = CodeAgent(
    tools=[
        check_hf_dataset_subset,
        dataset_download_tool,
        detect_problem_type_tool,
        map_row_tool,
        reformat_dataset_tool,
        validate_dataset_tool,
        save_dataset_tool,
        load_dataset_tool
    ],
    model=model,
    max_steps=args.max_steps,
)

# Run Full Pipeline
output = agent.run("""
Download and save the dataset "HAERAE-HUB/KMMLU".
Then open and use the saved file path.
Then detect the problem_type and reformat the dataset.
Then validate the reformatted dataset.
Finally save the reformatted dataset.
""")

print("Pipeline complete.")
print(output)
