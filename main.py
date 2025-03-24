from preprocessing.extract_text import extract_text
from ner.text_ner import text_ner

file_path = "Resume-Sample-2"

analyse = extract_text(f"{file_path}.pdf")
analyse2 = text_ner(analyse)

print(analyse)

