import pymupdf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def extract_text(pdf_path):
    # Open the PDF file
    doc = pymupdf.open(pdf_path)
    output = ""

    # Extract text from each page
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # Load the page
        output += page.get_text()  # Append the text to the accumulator

    doc.close()

    # Tokenize the text into words
    tokens = word_tokenize(output)

    # Define stopwords and punctuation to remove
    stopwords_nltk = set(stopwords.words('english'))
    remove_punctuation = {",", ":", ".", "/", '"', "\uf0a7", "'", "!", "?", ";", "(", ")", "[", "]", "{", "}", "•"}

    # Remove punctuation and stopwords
    cleaned_words = [
        word.lower() for word in tokens
        if word.lower() not in stopwords_nltk and word not in remove_punctuation
    ]

    return cleaned_words
