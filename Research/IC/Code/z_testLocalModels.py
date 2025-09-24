import os
import time
import torch
from typing import Tuple
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------- CONFIG ----------------
MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E"
USE_8BIT = False
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.8
TOP_P = 0.9
TOP_K = 50

# Load Hugging Face token from .env
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_FOLDER, ".env"))
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

if not HF_TOKEN:
    raise RuntimeError("❌ No Hugging Face API token found in .env (HUGGINGFACE_API_KEY).")

# Login
login(token=HF_TOKEN)

# ---------------- MODEL LOADING ----------------
def load_model(model_name: str, use_8bit: bool, device: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    print(f"🔄 Loading tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.pad_token = "[PAD]"

    print(f"🔄 Loading model '{model_name}' (8bit={use_8bit}) on {device}...")
    load_kwargs = {"device_map": "auto", "torch_dtype": torch.float16} if device == "cuda" else {
        "device_map": {"": "cpu"},
        "torch_dtype": torch.float32,
    }

    if use_8bit and device == "cuda":
        try:
            import bitsandbytes  # noqa: F401
            load_kwargs["load_in_8bit"] = True
        except ImportError:
            print("⚠️ bitsandbytes not installed; continuing without 8-bit.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=HF_TOKEN,
        **load_kwargs,
    )

    # Only resize embeddings if we added new tokens
    if tokenizer.pad_token == "[PAD]":
        model.resize_token_embeddings(len(tokenizer))
    
    return tokenizer, model


# ---------------- GENERATION ----------------
def generate_reply(model, tokenizer, prompt: str) -> Tuple[str, float]:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        t0 = time.time()
        generated = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        t1 = time.time()

    output_ids = generated[0][input_ids.shape[-1]:]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    return response, t1 - t0


# ---------------- MAIN ----------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        tokenizer, model = load_model(MODEL_NAME, USE_8BIT, device)
    except Exception as e:
        print(f"❌ Could not load model '{MODEL_NAME}': {e}")
        print("➡️ Make sure you accepted the license at:")
        print(f"   https://huggingface.co/{MODEL_NAME}")
        return

    model.eval()
    print("\n✅ Chat ready. Type a message and press Enter.")
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

        prompt = f"{history}\nUser: {user}\nAssistant:" if history else f"User: {user}\nAssistant:"
        reply, elapsed = generate_reply(model, tokenizer, prompt)

        history = prompt + " " + reply
        print(f"\nAssistant:\n{reply.strip()}\n(Generated in {elapsed:.2f}s)\n")


if __name__ == "__main__":
    main()