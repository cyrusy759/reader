import pandas as pd
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import random
from pathlib import Path

def prepare_data(filepath):
    """Convert your CSV with separate columns into spaCy training format"""
    df = pd.read_csv(filepath)
    training_data = []
    
    # Create artificial text by combining all columns
    for _, row in df.iterrows():
        # Generate synthetic text by joining all fields with spaces
        text_parts = []
        entities = []
        
        # Define entity order and labels
        columns_order = [
            ('names', 'PERSON'),
            ('job_title', 'JOB_TITLE'),
            ('company_name', 'COMPANY'),
            ('street_address', 'ADDRESS'),
            ('city_name', 'CITY'),
            ('state_code', 'STATE'),
            ('school_name', 'SCHOOL'),
            ('p_language', 'PROGRAMMING_LANGUAGE'),
            ('soft_skill', 'SKILL')
        ]
        
        current_pos = 0
        for col, label in columns_order:
            if pd.notna(row[col]):
                value = str(row[col])
                text_parts.append(value)
                
                # Add entity annotation
                start = current_pos
                end = current_pos + len(value)
                entities.append((start, end, label))
                
                # Add space between entities
                current_pos = end + 1
                text_parts.append(" ")
        
        # Combine into final text
        text = "".join(text_parts).strip()
        
        if entities:
            training_data.append((text, {"entities": entities}))
    
    return training_data

def train_model(training_data, output_dir="model_output", n_iter=30):
    """Train spaCy NER model without config file"""
    nlp = spacy.blank("en")
    
    # Add NER pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    # Add entity labels
    for _, annotations in training_data:
        for ent in annotations.get("entities", []):
            ner.add_label(ent[2])
    
    # Convert to DocBin
    doc_bin = DocBin()
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annotations["entities"]:
            span = doc.char_span(start, end, label=label)
            if span is not None:
                ents.append(span)
        doc.ents = ents
        doc_bin.add(doc)
    
    # Save training data
    Path(output_dir).mkdir(exist_ok=True)
    doc_bin.to_disk(f"{output_dir}/train.spacy")
    
    # Train the model
    nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(training_data)
        losses = {}
        for text, annotations in training_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], losses=losses)
        print(f"Iteration {itn}, Losses: {losses}")
    
    # Save model
    nlp.to_disk(f"{output_dir}/model")
    return nlp

# Usage
if __name__ == "__main__":
    # 1. Prepare data from your CSV
    training_data = prepare_data("ner/training.csv")
    
    # 2. Train the model
    nlp = train_model(training_data)
    
    # 3. Test the model
    test_text = "John Doe Python Google 123 Main St New York NY"
    doc = nlp(test_text)
    print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])