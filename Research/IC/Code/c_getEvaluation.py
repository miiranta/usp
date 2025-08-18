import os
import csv
import time
import openai
from dotenv import load_dotenv

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))

INPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "sentences")
OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "csvs")

load_dotenv(os.path.join(SCRIPT_FOLDER, '.env'))

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

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
    "gpt-4o"
]

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
            print("Model not recognized.")
            
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
                max_completion_tokens=1
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
    if not os.path.exists(INPUT_FOLDER):
        print("Input folder does not exist.")
        return
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
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
            
            output_file_path = os.path.join(OUTPUT_FOLDER, f"{model}_{text_file.date.replace('/', '')}.csv")
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
                    
                    if evaluation.grade == -2:
                        print("Some error occurred during evaluation (grade -2).")
                        return
                    
                    # Sleep to avoid rate limits
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

if __name__ == "__main__":
    main()
