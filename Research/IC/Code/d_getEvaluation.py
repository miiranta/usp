import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import csv
import time
import openai
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

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
openrouter_client = openai.OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

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
SUA RESPOSTA DEVE SER APENAS UMA LETRA, SEM QUALQUER OUTRO TEXTO

A FRASE É:
"""

MODELS = [
    # CLOSED -----------
    
    # OPENAI
    "openai/gpt-5",
    
    # ANTHROPIC
    "anthropic/claude-sonnet-4",
    
    # GOOGLE
    "google/gemini-2.5-pro",
    
    # XAI
    "x-ai/grok-4-fast:free",
    
    # OPEN -------------
    
    # OPENAI
    "openai/gpt-oss-120b", #"openai/gpt-oss-120b:free",
    
    # META
    "meta-llama/llama-4-maverick:free",
    
    # GOOGLE
    "google/gemma-3-27b-it:free",

    # MICROSOFT
    "microsoft/phi-4",
    
    # DEEPSEEK
    "deepseek/deepseek-chat-v3.1:free"
    
]

TIMER_MODELS = [
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-4-maverick:free",
    "google/gemma-3-27b-it:free",
    "x-ai/grok-4-fast:free",
    "deepseek/deepseek-chat-v3.1:free"
]

RETRIES = 10

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
    
        if model == "openai/gpt-5":
            self.evaluate_openrouter(model, 1024)
            self.string_grade_to_int()
            print(f" -> {self.grade}")   
        elif model == "openai/gpt-oss-120b":
            self.evaluate_openrouter(model, 512)
            self.string_grade_to_int()
            print(f" -> {self.grade}")
        elif model == "google/gemini-2.5-pro":
            self.evaluate_openrouter(model, 128)
            self.string_grade_to_int()
            print(f" -> {self.grade}")
        elif model == "google/gemma-3-27b-it:free":
            self.evaluate_openrouter(model, 8)
            self.string_grade_to_int()
            print(f" -> {self.grade}")
        elif model == "deepseek/deepseek-chat-v3.1:free":
            self.evaluate_openrouter(model, 4)
            self.string_grade_to_int()
            print(f" -> {self.grade}")
        else:
            self.evaluate_openrouter(model, 1)
            self.string_grade_to_int()
            print(f" -> {self.grade}")

    def evaluate_openrouter(self, model, max_tokens=1):
        free = False
        if ":free" in model:
            free = True
            model_without_free = model.replace(":free", "")
        else:
            model_without_free = model
        
        # Try free model first if applicable
        if free:
            print(f" - Trying free.")
            try:
                response = openrouter_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": PROMPT + self.sentence}],
                    max_tokens=max_tokens,
                )
                self.grade = response.choices[0].message.content.upper().replace('\n', '').replace('.', '').replace('<｜BEGIN▁OF▁SENTENCE｜>', '').strip()
                return
            except Exception as e:
                print(f"Error evaluating with {model}: {e}")
        
        # Then try paid model
        try:
            print(f" - Trying paid.")
            response = openrouter_client.chat.completions.create(
                model=model_without_free,
                messages=[{"role": "user", "content": PROMPT + self.sentence}],
                max_tokens=max_tokens,
            )
            self.grade = response.choices[0].message.content.upper().replace('\n', '').replace('.', '').replace('<｜BEGIN▁OF▁SENTENCE｜>', '').strip()
            return
        except Exception as e:
            print(f"Error evaluating with {model}: {e}")
            return
    
def _date_key(d):
    day, month, year = map(int, d.split('/'))
    return (year, month, day)

def main():
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

    for model in MODELS:
        for text_file in text_files:
            
            safe_model = model.replace(':free', '').replace('/', '-')
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
                    if model in TIMER_MODELS:
                        time.sleep(0.15)
                        
            with open(output_file_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f, delimiter="|")
                writer.writerow(["Date", "Model", "Grade", "Sentence"])
                for evaluation in evaluations:
                    writer.writerow([
                        evaluation.date,
                        evaluation.model.replace(':free', ''),
                        evaluation.grade,
                        evaluation.sentence
                    ])

if __name__ == "__main__":
    main()
