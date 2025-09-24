import time
import torch
import os
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from huggingface_hub import login

MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E"

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_FOLDER, ".env"))
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
login(token=HUGGINGFACE_API_KEY)

def load_model(model_name: str, use_8bit: bool, device: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    print(f"Loading tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True,
        token=HUGGINGFACE_API_KEY,
    )

    if tokenizer is None or not hasattr(tokenizer, "encode"):
        raise RuntimeError(
            "❌ Failed to load tokenizer. Make sure you accepted the model license and your token has access."
        )

    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print(f"Loading model '{model_name}' (8bit={use_8bit}) on {device}...")
    load_kwargs = {}
    if device.startswith("cuda"):
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = torch.float16
        if use_8bit:
            try:
                import bitsandbytes  # noqa: F401
                load_kwargs["load_in_8bit"] = True
            except Exception:
                print("⚠️ bitsandbytes not available; continuing without 8-bit.")
    else:
        load_kwargs["device_map"] = {"": "cpu"}
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=HUGGINGFACE_API_KEY,
        **load_kwargs,
    )

    model.resize_token_embeddings(len(tokenizer))

    if device == "cpu":
        model.to("cpu")

    return tokenizer, model

def generate_reply(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
) -> Tuple[str, float]:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(next(model.parameters()).device)

    with torch.no_grad():
        t0 = time.time()
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        t1 = time.time()

    output_ids = generated[0][input_ids.shape[-1]:]
    assistant_response = tokenizer.decode(output_ids, skip_special_tokens=True)
    return assistant_response, t1 - t0


def main() -> None:
    model_name = MODEL_NAME
    use_8bit = False

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    tokenizer, model = load_model(model_name, use_8bit, device)
    model.eval()

    print("\n✅ Interactive chat ready. Type your message and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    history = ""
    while True:
        try:
            user = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if user.strip().lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        if history:
            prompt = history + "\nUser: " + user + "\nAssistant:"
        else:
            prompt = "User: " + user + "\nAssistant:"

        assistant_response, elapsed = generate_reply(
            model,
            tokenizer,
            prompt,
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
        )

        history = prompt + " " + assistant_response

        print("\nAssistant:")
        print(assistant_response.strip())
        print(f"\n(Generated in {elapsed:.2f}s)\n")

if __name__ == "__main__":
    main()
