import spacy
from spacy.training import Example
from math import random

def train_model(train_data, output_dir):
    # Load a pre-trained model or start fresh
    nlp = spacy.load("en_core_web_lg")
    
    # Add new NER label if it doesn't exist
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    ner.add_label("ADDRESS")
    
    # Disable other pipeline components during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        
        for itn in range(30):  # Number of training iterations
            losses = {}
            random.shuffle(train_data)
            
            for text, annotations in train_data:
                example = Example.from_dict(nlp.make_doc(text), annotations)
                nlp.update([example], drop=0.5, losses=losses, sgd=optimizer)
            
            print(f"Iteration {itn}, Losses: {losses}")
    
    # Save the trained model
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")

# Alternative: Train using spaCy's CLI for better performance
# python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./dev.spacy