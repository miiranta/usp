import os
import re
import PyPDF2

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "atas")
OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "sentences")

SENTENCES_WHITELIST = [ # Select sentences that contain these phrases
]

SENTENCES_BLACKLIST = [ # Select sentences that should not be included, overrides the whitelisted phrases
]

def read_pdf_text(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)

            pages = list() 
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                pages.append(page_text)
 
            full_text = "\n".join(pages)
            
            # SPECIFIC FILTERS FOR THE PDF FILES
            
            # Convert br word to a space
            full_text = re.sub(r'\bbr\b', ' ', full_text, flags=re.IGNORECASE)
            
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
    sentences = []

    # Break text into sentences (separated by periods, exclamation marks, or question marks)
    page_sentences = re.split(r'(?<=[.!?]) +', text)
    sentences.extend(page_sentences)
        
    return sentences

def select_sentences(sentences):
    filtered_sentences = []
    
    if len(SENTENCES_WHITELIST) == 0:
        filtered_sentences = sentences
    else:
        for sentence in sentences:
            
            # Check if the sentence contains any whitelisted phrases
            if any(phrase.lower() in sentence.lower() for phrase in SENTENCES_WHITELIST):
                filtered_sentences.append(sentence)
            
    if len(SENTENCES_BLACKLIST) != 0:
        for sentence in filtered_sentences[:]:
            
            # Remove sentences that contain any blacklisted phrases
            if any(phrase.lower() in sentence.lower() for phrase in SENTENCES_BLACKLIST):
                filtered_sentences.remove(sentence)
            
    # Remove single words or single numbers 
    filtered_sentences = [s for s in filtered_sentences if len(s.split()) > 1]
        
    return filtered_sentences
        
def main():
    folders = [f for f in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, f))]
    
    total_phrases = 0
    
    if not folders:
        print("No folders found in the input directory.")
        return
    
    for folder in folders:
        print(f"Processing folder: {folder}")
        
        html_sentences = []
        pdf_sentences = []
        
        html_files = [f for f in os.listdir(os.path.join(INPUT_FOLDER, folder)) if f.endswith('.html')]
        if html_files:
            html_extracted = read_html_text(os.path.join(INPUT_FOLDER, folder, html_files[0]))
            html_trimmed = trim(html_extracted)
            html_sentences = break_into_sentences(html_trimmed)
            html_sentences = select_sentences(html_sentences)
            print(f" > Found {len(html_sentences)} sentences in HTML file.")
            
        pdf_files = [f for f in os.listdir(os.path.join(INPUT_FOLDER, folder)) if f.endswith('.pdf')]
        if pdf_files:
            pdf_extracted = read_pdf_text(os.path.join(INPUT_FOLDER, folder, pdf_files[0]))
            pdf_trimmed = trim(pdf_extracted)
            pdf_sentences = break_into_sentences(pdf_trimmed)
            pdf_sentences = select_sentences(pdf_sentences)
            print(f" > Found {len(pdf_sentences)} sentences in PDF file.")
        
        final_sentences = []
        if len(html_sentences) < len(pdf_sentences) * 0.5:
            final_sentences = pdf_sentences
            total_phrases += len(pdf_sentences)
            print(" > > Using PDF sentences.")
        else:
            final_sentences = html_sentences
            total_phrases += len(html_sentences)
            print(" > > Using HTML sentences.")
            
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
            
        output_file_path = os.path.join(OUTPUT_FOLDER, f"{folder}.txt")
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for sentence in final_sentences:
                output_file.write(sentence + "\n")
                
    print(f"Total phrases extracted: {total_phrases}")

if __name__ == "__main__":
    main()
