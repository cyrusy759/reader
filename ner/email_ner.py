import spacy
from spacy.matcher import Matcher

def email_ner(check_list):
    label = "EMAIL"
    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab)
    pattern = [
        {"TEXT": {"REGEX": r"^[a-zA-Z0-9._%+-]+$"}},   # username part
        {"TEXT": {"REGEX": r"^@$"}},                   # @ symbol
        {"TEXT": {"REGEX": r"^[a-zA-Z0-9.-]+$"}},      # domain part
        {"TEXT": {"REGEX": r"^\.[a-zA-Z]{2,}$"}}       # TLD
    ]

    matcher.add(label, [pattern])

    text = " ".join(check_list)
    check = nlp(text)
    
    matches = matcher(check)
    
    emails = []
    for match_id, start, end in matches:
        span = check[start:end]
        emails.append(span.text)
    return label, emails

test = email_ner(["bibibien", "@", "rabuda", ".com", "bruh", "123", "hams", "@", "gmao", ".com"])
print(test)