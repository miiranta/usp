# Ignore
OPEN_MODELS = [
    # OPEN -------------
    
    # META
    'meta-llama/Llama-4-Scout-17B-16E',
    #'meta-llama/Llama-3.2-3B',
    #'meta-llama/Llama-3.1-70B',
    #'meta-llama/Meta-Llama-3-70B',
    #'meta-llama/Llama-2-70b-hf',
    
    # GOOGLE
    #'google/gemma-3n-E4B',
    #'google/gemma-3-27b-pt',
    #'google/gemma-2-27b',
    #'google/gemma-7b',
    #'google/recurrentgemma-9b',

    # MICROSOFT
    #'microsoft/Phi-4-reasoning-plus',
    #'microsoft/phi-4',
    #'microsoft/phi-2',
    #'microsoft/phi-1_5',
    #'microsoft/phi-1',
    
    # OPENAI
    #'openai/gpt-oss-120b',
    #'openai/gpt-oss-20b',
    #'openai-community/gpt2-xl',
    #'openai-community/openai-gpt',
]

import time
from typing import Tuple

MODEL_NAME = 'meta-llama/Llama-4-Scout-17B-16E'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str, use_8bit: bool, device: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    print(f"Loading tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    if not hasattr(tokenizer, 'pad_token') or not hasattr(tokenizer, 'decode'):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
        except Exception:
            raise RuntimeError(f"Failed to load a valid tokenizer for '{model_name}'.")

    if getattr(tokenizer, 'pad_token', None) is None:
        if getattr(tokenizer, 'eos_token', None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            if hasattr(tokenizer, 'add_special_tokens'):
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            else:
                try:
                    tokenizer.pad_token = '[PAD]'
                    if not hasattr(tokenizer, 'pad_token_id'):
                        tokenizer.pad_token_id = None
                except Exception:
                    raise RuntimeError(f"Tokenizer for '{model_name}' doesn't support adding special tokens.")

    print(f"Loading model '{model_name}' (8bit={use_8bit}) on {device}...")
    load_kwargs = {}
    if device.startswith('cuda'):
        load_kwargs['device_map'] = 'auto'
        load_kwargs['torch_dtype'] = torch.float16
        if use_8bit:
            try:
                import bitsandbytes  # type: ignore
                load_kwargs['load_in_8bit'] = True
            except Exception:
                print("bitsandbytes not available, continuing without 8-bit.")
    else:
        load_kwargs['device_map'] = {'': 'cpu'}
        load_kwargs['torch_dtype'] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # resize token embeddings if we added pad token
    model.resize_token_embeddings(len(tokenizer))
    if device == 'cpu':
        model.to('cpu')

    return tokenizer, model


def generate_reply(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tuple[str, float]:
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
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
        )
        t1 = time.time()

    output_ids = generated[0][input_ids.shape[-1]:]
    assistant_response = tokenizer.decode(output_ids, skip_special_tokens=True)
    return assistant_response, t1 - t0


def main() -> None:
    model_name = MODEL_NAME
    use_8bit = False
    max_new_tokens = 256
    temperature = 0.8
    top_p = 0.9
    top_k = 50

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    tokenizer, model = load_model(model_name, use_8bit, device)
    model.eval()

    print('\nInteractive chat ready. Type your message and press Enter.')
    print("Type 'exit' or 'quit' to stop.\n")

    history = ''
    while True:
        try:
            user = input('You: ')
        except (KeyboardInterrupt, EOFError):
            print('\nExiting.')
            break

        if user.strip().lower() in ('exit', 'quit'):
            print('Goodbye.')
            break

        if history:
            prompt = history + '\nUser: ' + user + '\nAssistant:'
        else:
            prompt = 'User: ' + user + '\nAssistant:'

        assistant_response, elapsed = generate_reply(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        history = prompt + ' ' + assistant_response

        print('\nAssistant:')
        print(assistant_response.strip())
        print(f"\n(Generated in {elapsed:.2f}s)\n")

if __name__ == '__main__':
    main()
