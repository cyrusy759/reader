import spacy

def text_ner(analyse_set):
    nlp = spacy.load("en_core_web_sm")
    text = " ".join(analyse_set)
    doc = nlp(text)
    
    entity_dict = {}

    for entity in doc.ents:
        entity_type = entity.label_
        entity_text = entity.text

        if entity_type not in entity_dict:
            entity_dict[entity_type] = []
        entity_dict[entity_type].append(entity_text)

    return entity_dict