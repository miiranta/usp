import os
import time
import torch
from typing import Tuple
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

# ---------------- CONFIG ----------------
MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E"
USE_8BIT = False  # Set to True for memory savings if needed
MAX_NEW_TOKENS = 256

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
        # Attempt 1: AutoTokenizer with use_fast=True (this one works for this model)
        lambda: AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True,
            token=HF_TOKEN,
        ),
        # Attempt 2: Standard AutoTokenizer (fallback)
        lambda: AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True,
            token=HF_TOKEN,
        ),
        # Attempt 3: Explicit LlamaTokenizer (requires sentencepiece)
        lambda: LlamaTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True,
            token=HF_TOKEN,
        ),
    ]
    
    for i, attempt in enumerate(tokenizer_attempts, 1):
        try:
            print(f"  Attempt {i}: Loading tokenizer...")
            result = attempt()
            
            # Validate that we got a proper tokenizer object
            if result is None or isinstance(result, bool):
                print(f"  ❌ Attempt {i} returned invalid type: {type(result)}")
                continue
            
            # Check if it has essential tokenizer methods
            if not (hasattr(result, 'encode') and hasattr(result, 'decode')):
                print(f"  ❌ Attempt {i} returned object without tokenizer methods: {type(result)}")
                continue
                
            tokenizer = result
            print(f"  ✅ Successfully loaded tokenizer with attempt {i} (type: {type(tokenizer)})")
            break
        except Exception as e:
            print(f"  ❌ Attempt {i} failed: {e}")
            if i == len(tokenizer_attempts):
                raise RuntimeError(f"Failed to load tokenizer after {len(tokenizer_attempts)} attempts")

    # Configure pad token more safely
    print("🔄 Configuring tokenizer...")
    try:
        # Verify we have a valid tokenizer object
        if tokenizer is None or isinstance(tokenizer, bool) or not hasattr(tokenizer, 'pad_token'):
            raise RuntimeError(f"Invalid tokenizer object of type: {type(tokenizer)}")
        
        print(f"  Tokenizer type: {type(tokenizer)}")
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
        print(f"  ❌ Error: Could not configure tokenizer: {e}")
        print("  ➡️ This suggests a fundamental issue with tokenizer loading.")
        raise RuntimeError(f"Tokenizer configuration failed: {e}")

    print(f"🔄 Loading model '{model_name}' (8bit={use_8bit}) on {device}...")
    
    # Configure model loading parameters for speed optimization
    load_kwargs = {
        "trust_remote_code": True,
        "token": HF_TOKEN,
        "low_cpu_mem_usage": True,
    }
    
    # Try to enable flash attention if available
    try:
        import flash_attn
        load_kwargs["attn_implementation"] = "flash_attention_2"
        print("  ✅ Flash Attention 2 will be used for faster inference")
    except ImportError:
        print("  ℹ️ Flash Attention not available - using standard attention")
        load_kwargs["attn_implementation"] = "eager"
    
    if device == "cuda":
        load_kwargs.update({
            "device_map": "auto",
            "dtype": torch.float16,  # Use half precision for speed
        })
    else:
        load_kwargs.update({
            "device_map": {"": "cpu"},
            "dtype": torch.float32,
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
        
        # Optimize model for inference speed
        if device == "cuda":
            try:
                # Try to compile the model for faster inference (PyTorch 2.0+)
                print("  🔄 Optimizing model with torch.compile...")
                model = torch.compile(model, mode="reduce-overhead")
                print("  ✅ Model compiled for faster inference")
            except Exception as compile_error:
                print(f"  ⚠️ Could not compile model: {compile_error}")
        
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        # Try alternative loading approach
        print("  🔄 Trying alternative loading approach...")
        load_kwargs.pop("low_cpu_mem_usage", None)
        load_kwargs.pop("attn_implementation", None)
        if "load_in_8bit" in load_kwargs:
            load_kwargs.pop("load_in_8bit")
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        print("  ✅ Model loaded with alternative approach")

    # Check token embeddings alignment but don't resize automatically
    try:
        tokenizer_vocab_size = len(tokenizer)
        if hasattr(model, 'get_input_embeddings'):
            model_vocab_size = model.get_input_embeddings().num_embeddings
            print(f"  ℹ️ Tokenizer vocab size: {tokenizer_vocab_size}")
            print(f"  ℹ️ Model vocab size: {model_vocab_size}")
            
            if tokenizer_vocab_size != model_vocab_size:
                print(f"  ⚠️ Vocab size mismatch detected - using model's original size")
                print(f"  ➡️ This is normal for this model - no resizing needed")
    except Exception as e:
        print(f"  ⚠️ Warning: Could not check embeddings: {e}")

    return tokenizer, model


# ---------------- GENERATION ----------------
def generate_reply(model, tokenizer, prompt: str) -> Tuple[str, float]:
    try:
        print(f"🔄 Starting generation for prompt: '{prompt[:50]}...'")
        
        # Use the official recommended approach with messages format
        messages = [
            {"role": "user", "content": prompt},
        ]
        
        # Apply chat template as recommended in the docs
        inputs = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        )
        
        # Move inputs to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        print(f"  ✅ Applied chat template: {inputs['input_ids'].shape}")
        print(f"  ✅ Input device: {inputs['input_ids'].device}")
        print(f"  ✅ Model device: {next(model.parameters()).device}")
        
        print("  🔄 Starting model.generate()...")
        
        with torch.no_grad():
            t0 = time.time()
            
            # Try minimal parameters first
            try:
                outputs = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=10,  # Very small for testing
                    do_sample=False,  # Greedy decoding - most reliable
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                print(f"  ✅ Basic generation worked!")
                
            except Exception as gen_error:
                print(f"  ❌ Basic generation failed: {gen_error}")
                # Try even more minimal approach
                print("  🔄 Trying minimal generation...")
                outputs = model.generate(
                    inputs['input_ids'],
                    max_new_tokens=5,
                    do_sample=False,
                )
            
            t1 = time.time()
            print(f"  ✅ Generation completed in {t1-t0:.2f}s")

        # Extract only the new tokens (after the input)
        output_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        print(f"  ✅ Generated {len(output_ids)} new tokens: {output_ids}")
        
        # Decode using batch_decode as shown in docs
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        print(f"  ✅ Decoded response: '{response}'")
        
        return response, t1 - t0
    
    except Exception as e:
        print(f"❌ Generation error: {e}")
        import traceback
        traceback.print_exc()
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