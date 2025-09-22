import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import csv
import time
import openai
import torch
import unicodedata
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))

INPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "sentences_selected")
OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "csvs")

if not os.path.exists(INPUT_FOLDER):
    print("Input folder does not exist.")
    exit(1)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

load_dotenv(os.path.join(SCRIPT_FOLDER, '.env'))

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

loaded_model = None
loaded_tokenizer = None

PROMPT = """
DEFINIÇÃO DE OTIMISMO:
Ocorre quando as projeções indicam que a inflação ficará abaixo da meta ou dentro do intervalo de tolerância com folga. 
Isso pode sinalizar que o Banco Central vê espaço para reduzir juros ou manter uma política monetária mais acomodatícia. 

DEFINIÇÃO DE PESSIMISMO:
Ocorre quando as projeções apontam para inflação acima da meta ou próxima do teto do intervalo de tolerância. 
Isso sugere preocupação com pressões inflacionárias e pode justificar uma política monetária mais restritiva.

AVALIE A FRASE COMO
O para OTIMISTA
N para NEUTRA
P para PESSIMISTA
SUA RESPOSTA DEVE SER APENAS UMA LETRA, SEM QUALQUER OUTRO TEXTO.

A FRASE É:
"""

MODELS = [
    # CLOSED -----------
    
    # OPENAI
    #"gpt-3.5-turbo",
    #"gpt-4o",
    #"gpt-5",
]

OPEN_MODELS = [
    # OPEN -------------
    
    # META
    'meta-llama/Llama-3.2-3B',
    'meta-llama/Llama-4-Scout-17B-16E',
    'meta-llama/Llama-3.1-70B',
    'meta-llama/Meta-Llama-3-70B',
    'meta-llama/Llama-2-70b-hf',
    
    # GOOGLE
    'google/gemma-3n-E4B',
    'google/gemma-3-27b-pt',
    'google/gemma-2-27b',
    'google/gemma-7b',
    'google/recurrentgemma-9b',

    # MICROSOFT
    'microsoft/Phi-4-reasoning-plus',
    'microsoft/phi-4',
    'microsoft/phi-2',
    'microsoft/phi-1_5',
    'microsoft/phi-1',
    
    # OPENAI
    'openai/gpt-oss-120b',
    'openai/gpt-oss-20b',
    'openai-community/gpt2-xl',
    'openai-community/openai-gpt',
]

RETRIES = 3

class File:
    def __init__(self, file, date):
        self.file = file
        self.date = date

class Evaluation:
    def __init__(self, date, sentence):
        self.date = date
        self.sentence = sentence
        self.model = ""
        self.grade = -2
        
    def string_grade_to_int(self):
        if self.grade == "O":
            self.grade = 1
        elif self.grade == "N":
            self.grade = 0
        elif self.grade == "P":
            self.grade = -1
        else:
            print(f"Unexpected grade: {self.grade}. Setting to -2.")
            self.grade = -2

    def evaluate(self, model):
        self.model = model
        
        if model == "gpt-3.5-turbo":
            self.evaluate_openai(model)
            self.string_grade_to_int()
            print(f" -> {self.grade}")
            
        elif model == "gpt-4o":
            self.evaluate_openai(model)
            self.string_grade_to_int()
            print(f" -> {self.grade}")
            
        elif model == "gpt-5":
            self.evaluate_openai_gpt5(model)
            self.string_grade_to_int()
            print(f" -> {self.grade}")
            
        else:
            self.evaluate_open_model(model)
            self.string_grade_to_int()
            print(f" -> {self.grade}")
           
    def evaluate_open_model(self, model):
        global loaded_model, loaded_tokenizer
        if loaded_model is None or loaded_tokenizer is None:
            print("No model or tokenizer loaded.")
            self.grade = -2
            return
        try:
            norm = unicodedata.normalize('NFKC', self.sentence)
            prompt_with_input = PROMPT + norm + "\nRESPOSTA:"
            inputs = loaded_tokenizer(prompt_with_input, return_tensors="pt")
            with torch.no_grad():
                generated = loaded_model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=5000,
                    pad_token_id=loaded_tokenizer.eos_token_id,
                )

            decoded = loaded_tokenizer.decode(generated[0], skip_special_tokens=True).upper().strip()
            sanitized = decoded.replace('\r', ' ').replace('\n', ' ').strip()
            print(f' -->: {sanitized[-15:]}')

            for ch in reversed(sanitized):
                if ch in ("O", "N", "P"):
                    self.grade = ch
                    return

            return
        except Exception as e:
            print(f"Error evaluating with {model}: {e}")
            self.grade = -2
            return
     
    def evaluate_openai(self, model):
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT + self.sentence}],
                max_tokens=1
            )
            self.grade = response.choices[0].message.content.upper()
            return
        except Exception as e:
            print(f"Error evaluating with {model}: {e}")
            return
        
    def evaluate_openai_gpt5(self, model):
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT + self.sentence}],
                max_completion_tokens=5000
            )
            self.grade = response.choices[0].message.content.upper()
            return
        except Exception as e:
            print(f"Error evaluating with {model}: {e}")
            return

def _date_key(d):
    day, month, year = map(int, d.split('/'))
    return (year, month, day)

def main():
    global loaded_model, loaded_tokenizer
    
    raw_text_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.txt')]
    if not raw_text_files:
        print("No text files found in the input directory.")
        return
    
    text_files = []
    for text_file in raw_text_files:
        date = text_file[:-4]
        date = f"{date[:2]}/{date[2:4]}/{date[4:8]}"
        text_files.append(File(text_file, date))
        
    text_files.sort(key=lambda f: _date_key(f.date))

    for model in MODELS + OPEN_MODELS:
        
        # Open model? Load it
        if model in OPEN_MODELS:
            print(f"Loading model {model}...")

            original_cuda_available = torch.cuda.is_available
            original_get_device_capability = torch.cuda.get_device_capability
            original_get_device_properties = torch.cuda.get_device_properties
            torch.cuda.is_available = lambda: False
            torch.cuda.get_device_capability = lambda device=None: (0, 0)
            torch.cuda.get_device_properties = lambda device: None
            
            try:
                loaded_model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=model,
                    device_map="cpu",
                    dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    use_safetensors=True,
                    attn_implementation="eager",
                )
                
                loaded_tokenizer = AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=model,
                    trust_remote_code=True,
                )
                
                print(f"Successfully loaded {model}")
                
            except Exception as e:
                print(f"Error loading model {model}: {e}")
                print("Skipping this model...")
                continue
                
            finally:
                torch.cuda.is_available = original_cuda_available
                torch.cuda.get_device_capability = original_get_device_capability
                torch.cuda.get_device_properties = original_get_device_properties
                torch.cuda.empty_cache()
                
        for text_file in text_files:
            
            safe_model = model.replace('/', '_')
            output_file_path = os.path.join(OUTPUT_FOLDER, f"{safe_model}_{text_file.date.replace('/', '')}.csv")
            if os.path.exists(output_file_path):
                print(f"Output file for {model} on {text_file.date} already exists. Skipping...")
                continue
            
            evaluations = []
            with open(os.path.join(INPUT_FOLDER, text_file.file), 'r', encoding='utf-8') as f:
                sentences = f.readlines()
            
            sentences_amount = len(sentences)
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if sentence:
                    print(f"[{text_file.date} | {model}] {i + 1}/{sentences_amount}: {sentence}")
                    
                    evaluation = Evaluation(text_file.date, sentence)
                    evaluation.evaluate(model)
                    evaluations.append(evaluation)
                    
                    retries_left = RETRIES
                    if retries_left > 0 and evaluation.grade == -2:
                        while retries_left > 0 and evaluation.grade == -2:
                            print("Some error occurred during evaluation (grade -2).")
                            print(f"Retrying... {retries_left} retries left.")
                            evaluation.evaluate(model)
                            if evaluation.grade == -2:
                                retries_left -= 1
                            
                        if retries_left == 0:
                            print("Max retries reached. Stopping.")
                            return
                                
                    # Sleep to avoid rate limits
                    if model not in OPEN_MODELS:
                        time.sleep(0.15)
                        
            with open(output_file_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f, delimiter="|")
                writer.writerow(["Date", "Model", "Grade", "Sentence"])
                for evaluation in evaluations:
                    writer.writerow([
                        evaluation.date,
                        evaluation.model,
                        evaluation.grade,
                        evaluation.sentence
                    ])
                    
        # Unload model
        if model in OPEN_MODELS:
            del loaded_model
            del loaded_tokenizer
            torch.cuda.empty_cache()
            print(f"Unloaded model {model}.")

if __name__ == "__main__":
    main()
