import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_TO_READ = os.path.join(SCRIPT_DIR, "atas/01022023.html")

SENTENCES_WHITELIST = [
    "inflação",
]

def read_html_text(html_path):
    try:
        with open(html_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            
            # Get only body content
            body_content = re.search(r'<body[^>]*>(.*?)</body>', file_content, re.DOTALL)
            if not body_content:
                print("No body content found in HTML.")
                return None
            
            # Remove all a tags and its content
            body_text = re.sub(r'<a[^>]*>.*?</a>', ' ', body_content.group(1), flags=re.DOTALL)
            
            # Convert anything <...> to a .
            body_text = re.sub(r'<[^>]+>', '.', body_text)
            
            # Convert multiple dots to a single dot
            body_text = re.sub(r'\.{2,}', '.', body_text)
            
            return body_text

    except Exception as e:
        print(e)
        return None
    
def trim(text):

    # Remove newlines and tabs
    cleaned_text = re.sub(r'\n+', ' ', text)
    cleaned_text = re.sub(r'\t+', ' ', cleaned_text)
    
    # Replace multiple spaces with single space
    cleaned_text = re.sub(r' +', ' ', cleaned_text)
    
    # Strip leading and trailing whitespace
    cleaned_text = cleaned_text.strip()
    
    # Add a space after punctuation if it doesn't exist
    cleaned_text = re.sub(r'([.!?;,:])(?=\S)', r'\1 ', cleaned_text)
    
    # Remove whitespace and newlines before .,!?,:;
    cleaned_text = re.sub(r'\s*([.!?,:;])', r'\1', cleaned_text)
    
    # Normalize dashes: remove all whitespace around dashes
    cleaned_text = re.sub(r'\s*-\s*', '-', cleaned_text)
    
    return cleaned_text

def break_into_sentences(text):
    sentences = []

    # Break text into sentences (separated by periods, exclamation marks, or question marks)
    page_sentences = re.split(r'(?<=[.!?;]) +', text)
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
    
    extracted = read_html_text(FILE_TO_READ)
    trimmed = trim(extracted)
    sentences = break_into_sentences(trimmed)
    sentences = select_sentences(sentences)
    for sentence in sentences:
        print(sentence)
        print()
        
    print("----")
    print("Total sentences found:", len(sentences))
    print("----")

if __name__ == "__main__":
    main()
