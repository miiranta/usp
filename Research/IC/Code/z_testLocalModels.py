import os
import time
import torch
from typing import Tuple
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

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
    
    # Try multiple approaches for tokenizer loading
    tokenizer = None
    tokenizer_attempts = [
        # Attempt 1: Standard AutoTokenizer
        lambda: AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True,
            token=HF_TOKEN,
        ),
        # Attempt 2: Explicit LlamaTokenizer
        lambda: LlamaTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True,
            token=HF_TOKEN,
        ),
        # Attempt 3: AutoTokenizer with use_fast=True
        lambda: AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True,
            token=HF_TOKEN,
        ),
    ]
    
    for i, attempt in enumerate(tokenizer_attempts, 1):
        try:
            print(f"  Attempt {i}: Loading tokenizer...")
            tokenizer = attempt()
            print(f"  ✅ Successfully loaded tokenizer with attempt {i}")
            break
        except Exception as e:
            print(f"  ❌ Attempt {i} failed: {e}")
            if i == len(tokenizer_attempts):
                raise RuntimeError(f"Failed to load tokenizer after {len(tokenizer_attempts)} attempts")

    # Configure pad token more safely
    print("🔄 Configuring tokenizer...")
    try:
        # Verify we have a valid tokenizer object
        if tokenizer is None or not hasattr(tokenizer, 'pad_token'):
            raise RuntimeError("Invalid tokenizer object")
        
        print(f"  Current pad_token: {getattr(tokenizer, 'pad_token', 'None')}")
        print(f"  Current eos_token: {getattr(tokenizer, 'eos_token', 'None')}")
        
        # Check if tokenizer has necessary attributes
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                print("  ✅ Set pad_token to eos_token")
            elif hasattr(tokenizer, 'bos_token') and tokenizer.bos_token is not None:
                tokenizer.pad_token = tokenizer.bos_token
                print("  ✅ Set pad_token to bos_token")
            else:
                # Last resort - add a new pad token
                try:
                    num_added = tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                    if num_added > 0:
                        print(f"  ✅ Added new pad token, {num_added} tokens added")
                    else:
                        print("  ℹ️ Pad token already exists")
                except Exception as add_token_error:
                    print(f"  ⚠️ Could not add pad token: {add_token_error}")
                    # Set a default pad token manually
                    tokenizer.pad_token = "<pad>"
                    print("  ✅ Set pad_token to <pad> manually")
        else:
            print("  ✅ Pad token already configured")
        
        # Ensure pad_token_id is set
        if tokenizer.pad_token_id is None and tokenizer.pad_token is not None:
            try:
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
                print(f"  ✅ Set pad_token_id to {tokenizer.pad_token_id}")
            except Exception as convert_error:
                print(f"  ⚠️ Could not set pad_token_id: {convert_error}")
    
    except Exception as e:
        print(f"  ⚠️ Warning: Could not configure pad token: {e}")
        print(f"  ⚠️ Tokenizer type: {type(tokenizer)}")
        # Continue anyway, some models work without explicit pad tokens

    print(f"🔄 Loading model '{model_name}' (8bit={use_8bit}) on {device}...")
    
    # Configure model loading parameters
    load_kwargs = {
        "trust_remote_code": True,
        "token": HF_TOKEN,
        "low_cpu_mem_usage": True,  # Help with memory efficiency
    }
    
    if device == "cuda":
        load_kwargs.update({
            "device_map": "auto",
            "dtype": torch.float16,  # Updated from torch_dtype
        })
    else:
        load_kwargs.update({
            "device_map": {"": "cpu"},
            "dtype": torch.float32,  # Updated from torch_dtype
        })

    # Add quantization if requested
    if use_8bit and device == "cuda":
        try:
            import bitsandbytes  # noqa: F401
            load_kwargs["load_in_8bit"] = True
            print("  ✅ 8-bit quantization enabled")
        except ImportError:
            print("  ⚠️ bitsandbytes not installed; continuing without 8-bit.")

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        print("  ✅ Model loaded successfully")
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        # Try alternative loading approach
        print("  🔄 Trying alternative loading approach...")
        load_kwargs.pop("low_cpu_mem_usage", None)
        if "load_in_8bit" in load_kwargs:
            load_kwargs.pop("load_in_8bit")
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        print("  ✅ Model loaded with alternative approach")

    # Resize token embeddings if necessary
    try:
        original_vocab_size = len(tokenizer)
        if hasattr(model, 'get_input_embeddings'):
            current_vocab_size = model.get_input_embeddings().num_embeddings
            if original_vocab_size != current_vocab_size:
                print(f"  🔄 Resizing token embeddings from {current_vocab_size} to {original_vocab_size}")
                model.resize_token_embeddings(original_vocab_size)
                print("  ✅ Token embeddings resized")
    except Exception as e:
        print(f"  ⚠️ Warning: Could not resize embeddings: {e}")

    return tokenizer, model


# ---------------- GENERATION ----------------
def generate_reply(model, tokenizer, prompt: str) -> Tuple[str, float]:
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        
        # Get attention mask if available
        attention_mask = None
        if "attention_mask" in inputs:
            attention_mask = inputs.attention_mask.to(model.device)

        with torch.no_grad():
            t0 = time.time()
            
            # Prepare generation arguments
            gen_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": True,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
            }
            
            # Add optional parameters if available
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask
            
            if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
            
            if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
            
            generated = model.generate(**gen_kwargs)
            t1 = time.time()

        # Extract only the new tokens
        output_ids = generated[0][input_ids.shape[-1]:]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        return response, t1 - t0
    
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return f"[Error during generation: {e}]", 0.0


# ---------------- MAIN ----------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")

    try:
        tokenizer, model = load_model(MODEL_NAME, USE_8BIT, device)
    except Exception as e:
        print(f"❌ Could not load model '{MODEL_NAME}': {e}")
        print("➡️ Troubleshooting suggestions:")
        print(f"   1. Make sure you accepted the license at: https://huggingface.co/{MODEL_NAME}")
        print("   2. Verify your HUGGINGFACE_API_KEY is correct")
        print("   3. Try using the Instruct version: meta-llama/Llama-4-Scout-17B-16E-Instruct")
        print("   4. Check if you have enough GPU memory (model is ~17B parameters)")
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

        # Build prompt with history
        if history:
            prompt = f"{history}\nUser: {user}\nAssistant:"
        else:
            prompt = f"User: {user}\nAssistant:"
        
        # Generate response
        reply, elapsed = generate_reply(model, tokenizer, prompt)
        
        # Update history
        history = prompt + " " + reply
        
        # Truncate history if it gets too long (to prevent context overflow)
        if len(history) > 4000:  # Rough character limit
            history = "..." + history[-3000:]  # Keep last 3000 chars
        
        print(f"\nAssistant:\n{reply.strip()}\n(Generated in {elapsed:.2f}s)\n")


if __name__ == "__main__":
    main()