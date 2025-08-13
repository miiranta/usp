import PyPDF2
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_TO_READ = os.path.join(SCRIPT_DIR, "Copom272.pdf")

SENTENCES_WHITELIST = [
    "inflação",
]

def read_pdf_text(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)

            full_text = list() 
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                full_text.append(page_text)
 
            return full_text
        
    except Exception as e:
        print(e)
        return None
    
def trim(pages):
    trimmed_pages = []
    for page in pages:

        # Remove newlines and tabs
        cleaned_text = re.sub(r'\n+', ' ', page)
        cleaned_text = re.sub(r'\t+', ' ', cleaned_text)
        
        # Replace multiple spaces with single space
        cleaned_text = re.sub(r' +', ' ', cleaned_text)
        
        # Strip leading and trailing whitespace
        cleaned_text = cleaned_text.strip()
        
        # Remove whitespace and newlines before .,!?,:;
        cleaned_text = re.sub(r'\s*([.!?,:;])', r'\1', cleaned_text)
        
        # Normalize dashes: remove all whitespace around dashes
        cleaned_text = re.sub(r'\s*-\s*', '-', cleaned_text)
        
        trimmed_pages.append(cleaned_text)
    
    return trimmed_pages

def break_into_sentences(pages):
    sentences = []
    for page in pages:
        
        # Break text into sentences (separated by periods, exclamation marks, or question marks)
        page_sentences = re.split(r'(?<=[.!?]) +', page)
        sentences.extend(page_sentences)
        
    return sentences

def select_sentences(sentences):
    filtered_sentences = []
    for sentence in sentences:
        
        # Check if the sentence contains any whitelisted phrases
        # Should not consider spaces or punctuation
        # Should not be case-sensitive
        if any(phrase.lower() in sentence.lower() for phrase in SENTENCES_WHITELIST):
            filtered_sentences.append(sentence)
        
    return filtered_sentences
        
def main():
    if not os.path.exists(FILE_TO_READ):
        print(f"File '{FILE_TO_READ}' does not exist.")
        return
    
    extracted_pages = read_pdf_text(FILE_TO_READ)
    trimmed_pages = trim(extracted_pages)
    sentences = break_into_sentences(trimmed_pages)
    sentences = select_sentences(sentences)
    for sentence in sentences:
        print(sentence)
        print()
        
    print("----")
    print("Total sentences found:", len(sentences))
    print("----")

if __name__ == "__main__":
    main()
