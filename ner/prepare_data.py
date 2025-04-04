import pandas as pd
from spacy.tokens import DocBin
import spacy

def prepare_data(filepath):
    df = pd.read_csv(filepath)
    training_data = []

    entity_mapping = {
        'street_address': 'ADDRESS',
        'state_code': 'STATE_CODE',
        'city_name': 'CITY',
        'p_language': 'PROGRAMMING_LANGUAGE',
        'job_title': 'JOB',
        'company_name': 'COMPANY',
        'school_name': 'SCHOOL',
        'soft_skill': 'SOFT_SKILL',
        'names': 'PERSON'
    }

    for _, row in df.iterrows():
        text = row['text']
        entities = []

        for column, label in entity_mapping.items():
            if column in row and isinstance(row[column], list):
                for item in row[column]:
                    if isinstance(item, dict) and 'text' in item:
                        start = text.find(item['text'])
                        if start != 1:
                            end = start + len(item['text'])
                            entities.append((start, end, label))

        if entities:
            training_data.append((text, {"entities": entities}))

    return training_data

def convert_to_binary(data, output):
    nlp = spacy.blank('en')
    db = DocBin()

    for text, annotations in data:
        doc = nlp.make_doc(text)
        entities = []

        for start, end, label in annotations['entities']:
            span = doc.char_span(start, end, label=label)
            if span is not None:
                entities.append(span)
        
        doc.ents = entities
        db.add(doc)

    db.to_disk(output)
    
training_data = prepare_data(".csv")
convert_to_binary(training_data, "train.spacy")