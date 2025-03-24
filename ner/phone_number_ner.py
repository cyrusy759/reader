import spacy
from spacy.matcher import Matcher

def phone_number_ner(check_list):
    label = "PHONE_NUMBER"
    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab)
    pattern_1 = [{"ORTH": "("}, {"SHAPE": "ddd"}, {"ORTH": ")"}, {"ORTH": " ", "OP": "?"}, {"SHAPE": "ddd"},
            {"ORTH": "-", "OP": "?"}, {"SHAPE": "ddd"}]
    pattern_2 = [{"SHAPE": "ddd"}, {"ORTH": "-", "OP": "?"}, {"SHAPE": "ddd"}, {"ORTH": "-", "OP": "?"}, {"SHAPE": "ddd"}]

    matcher.add(label, [pattern_1])
    matcher.add(label, [pattern_2])

    text = " ".join(check_list)
    check = nlp(text)
    
    matches = matcher(check)
    
    phone_numbers = []
    for match_id, start, end in matches:
        span = check[start:end]
        phone_numbers.append(span.text)
    return label, phone_numbers
    print(phone_numbers)

phone_number_ner({"998-345-997", "no", "887-998-124"})