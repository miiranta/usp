import os
import re
import math
import subprocess
import spacy
from spacy_layout import spaCyLayout
from pdfminer.high_level import extract_text

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "atas")
OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "sentences")

if not os.path.exists(INPUT_FOLDER):
    print("Input folder does not exist.")
    exit(1)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
   
SIGMA_THRESHOLD = 2
SIGMA_OFFSET = 2
    
BLACKLIST = [
    "javascript",
    "cookies",
    "expand_less",
    "content_copy",
    "Garantir a estabilidade do poder de compra da moeda,",
]

try:
    nlp = spacy.load("pt_core_news_lg")
    layout = spaCyLayout(nlp)
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "pt_core_news_lg"])
    nlp = spacy.load("pt_core_news_lg")
    layout = spaCyLayout(nlp)

def read_pdf_text(pdf_path):
    try:
        doc = layout(pdf_path)
        full_text = doc.text
         
        # SPECIFIC FILTERS FOR THE PDF FILES
        # ---
        
        return full_text
        
    except Exception as e:
        return ""

def read_html_text(html_path):
    try:
        with open(html_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            
            # SPECIFIC FILTERS FOR THE HTML FILES
            
            # Get only body content
            body_content = re.search(r'<body[^>]*>(.*?)</body>', file_content, re.DOTALL)
            if not body_content:
                print("No body content found in HTML.")
                return None
            
            # Remove all a tags and its content
            body_text = re.sub(r'<a[^>]*>.*?</a>', ' ', body_content.group(1), flags=re.DOTALL)
            
            # Strong, i, br tags are converted to a space
            body_text = re.sub(r'<strong[^>]*>.*?</strong>', ' ', body_text, flags=re.DOTALL)
            body_text = re.sub(r'<br[^>]*>', ' ', body_text, flags=re.DOTALL)
            body_text = re.sub(r'<i[^>]*>.*?</i>', ' ', body_text, flags=re.DOTALL)
            
            # Convert anything <...> to a .
            body_text = re.sub(r'<[^>]+>', '.', body_text)
            
            # Convert multiple dots to a single dot
            body_text = re.sub(r'\.{2,}', '.', body_text)
            
            # ---
            
            return body_text

    except Exception as e:
        return ""
    
def trim(text):
    # Remove newlines, tabs and &nbsp;
    cleaned_text = re.sub(r'\n+', ' ', text)
    cleaned_text = re.sub(r'\t+', ' ', cleaned_text)
    cleaned_text = re.sub(r'&nbsp;', ' ', cleaned_text)
    
    # Replace multiple spaces with single space
    cleaned_text = re.sub(r' +', ' ', cleaned_text)
    
    # Strip leading and trailing whitespace
    cleaned_text = cleaned_text.strip()
    
    # Add a space after punctuation if it doesn't exist
    cleaned_text = re.sub(r'([.!?;,:])(?=\S)', r'\1 ', cleaned_text)
    
    # (space)(single letter)(dot) becomes a space.
    cleaned_text = re.sub(r'(\s)[A-Za-z]\.', r'\1', cleaned_text)
    
    # Remove whitespace and newlines before .,!?,:;
    cleaned_text = re.sub(r'\s*([.!?,:;])', r'\1', cleaned_text)
    
    # Normalize dashes: remove all whitespace around dashes
    cleaned_text = re.sub(r'\s*-\s*', '-', cleaned_text)
    
    # A bunch of dots become a single dot
    cleaned_text = re.sub(r'\.{2,}', '.', cleaned_text)
    
    return cleaned_text

def break_into_sentences(text):
    doc = nlp(text)
    cleaned_sentences = []
    
    for sent in doc.sents:
        cleaned_sentences.append(sent.text.strip())
        
    return cleaned_sentences

def trim_phrases(phrases):
    sigma_threshold = SIGMA_THRESHOLD
    sigma_offset = SIGMA_OFFSET
    
    cleaned_phrases = []

    # Remove phares that do not end in ., !, ?
    for phrase in phrases:
        if re.search(r'[.!?]$', phrase):
            cleaned_phrases.append(phrase)
    
    # Remove one word phrases
    cleaned_phrases = [phrase for phrase in cleaned_phrases if len(phrase.split()) > 1]

    # Remove phrases in blacklist
    for word in BLACKLIST:
        cleaned_phrases = [phrase for phrase in cleaned_phrases if word.lower() not in phrase.lower()]
        
    # Remove phrases that are too short
    lengths = [len(phrase) for phrase in cleaned_phrases]
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    sd = math.sqrt(variance)
    mean_corrected = sum(lengths) / len(lengths) + sigma_offset * sd
    lower_threshold = mean_corrected - sigma_threshold * sd
    cleaned_phrases = [phrase for phrase in cleaned_phrases if len(phrase) >= lower_threshold]
    
    return cleaned_phrases

def main():
    folders = [f for f in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, f))]
    
    total_phrases = 0
    
    if not folders:
        print("No folders found in the input directory.")
        return
    
    for folder in folders:
        print(f"Processing folder: {folder}")
        
        pdf_sentences_final = []
        pdf_files = [f for f in os.listdir(os.path.join(INPUT_FOLDER, folder)) if f.endswith('.pdf')]
        if pdf_files:
            pdf_extracted = read_pdf_text(os.path.join(INPUT_FOLDER, folder, pdf_files[0]))
            pdf_trimmed = trim(pdf_extracted)
            pdf_sentences = break_into_sentences(pdf_trimmed)
            pdf_sentences_final = trim_phrases(pdf_sentences)
            print(f" > Found {len(pdf_sentences_final)} sentences in PDF file.")
        
        html_sentences_final = []
        if len(pdf_sentences_final) == 0:
            html_files = [f for f in os.listdir(os.path.join(INPUT_FOLDER, folder)) if f.endswith('.html')]
            if html_files:
                html_extracted = read_html_text(os.path.join(INPUT_FOLDER, folder, html_files[0]))
                html_trimmed = trim(html_extracted)
                html_sentences = break_into_sentences(html_trimmed)
                html_sentences_final = trim_phrases(html_sentences)
                print(f" > Found {len(html_sentences_final)} sentences in HTML file.")
        
        final_sentences = []
        if len(html_sentences_final) < len(pdf_sentences_final):
            final_sentences = pdf_sentences_final
            total_phrases += len(pdf_sentences_final)
            print(" > > Using PDF sentences.")
        else:
            final_sentences = html_sentences_final
            total_phrases += len(html_sentences_final)    
            print(" > > Using HTML sentences.")
              
        output_file_path = os.path.join(OUTPUT_FOLDER, f"{folder}.txt")
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for sentence in final_sentences:
                output_file.write(sentence + "\n")
                
    print(f"Total phrases extracted: {total_phrases}")

if __name__ == "__main__":
    main()
