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
EVALUATE THE SENTENCE AS
O for OPTIMISTIC
N for NEUTRAL
P for PESSIMISTIC
YOUR ANSWER MUST BE ONLY ONE LETTER, WITHOUT ANY OTHER TEXT.

THE SENTENCE IS:
"""

MODELS = [
    "gpt-3.5-turbo",
    "gpt-4o"
]

class File:
    def __init__(self, file, date):
        self.file = file
        self.date = date
        
    def order_time_s(self):
        day, month, year = map(int, self.date.split('/'))
        return (year * 365 + month * 30 + day) * 24 * 60 * 60

class Evaluation:
    def __init__(self, date, sentence):
        self.date = date
        self.sentence = sentence
        self.model = ""
        self.grade = 0
        
    def order_time_s(self):
        day, month, year = map(int, self.date.split('/'))
        return (year * 365 + month * 30 + day) * 24 * 60 * 60

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
        
    text_files.sort(key=lambda f: f.order_time_s())

    for model in MODELS:

        evaluations = []
        for text_file in text_files:
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
                    
                    # Sleep to avoid rate limits
                    time.sleep(0.15)
                        
        evaluations.sort(key=lambda e: (e.model, e.order_time_s()))
        output_file_path = os.path.join(OUTPUT_FOLDER, f"{model}.csv")

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
