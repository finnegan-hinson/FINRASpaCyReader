# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:03:07 2018

@author: Virginia Hinson

Python beginning to use Named-entity recognition with spaCy
"""

from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
import re as standardre

from pdfMiner import convert_pdf_to_txt

# training data
TRAIN_DATA = [
    ('Gates Capital Corporation (CRD® #29582, New York, New York), ', {
        'entities': [(0, 25, 'ORG'), (40, 48, 'LOC'), (50, 58, 'LOC')]
    }),
    ('James Douglas Casey III (CRD #500062, Greenwich, Connecticut), John Charles Fitzgerald  (CRD #1529631, Lake Arrow Head, California) and Youngwhi Kim  (CRD #1394474, Hartsdale, New York) '
    , {
        'entities': [(0, 23, 'PERSON'), (38, 47, 'LOC'), (49, 60, 'LOC'),
                     (63, 86, 'PERSON'), (103, 118, 'LOC'), (120, 130, 'LOC'),
                     (113, 125, 'PERSON'), (142, 151, 'LOC'), (153, 161, 'LOC')]
    }),
    
    ('ACP Securities, LLC (CRD #139049, Miami, Florida), ', {
        'entities': [(0, 19, 'ORG')]
    }),

    ('Merrill Lynch, Pierce, Fenner & Smith Incorporated (CRD #7691, New York, New York), ', {
        'entities': [(0,  50, 'ORG')]
      }),
    ('Allen & Company of Florida, Inc. (CRD #25, Lakeland, Florida), ', {
        'entities': [(0, 32, 'ORG')]
      })
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model= "xx_ent_wiki_sm", output_dir=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")



    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
        print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    print()    
    
    new_data = convert_pdf_to_txt('C:\\Users\\vrh\\Downloads\\july_2018_Disciplinary_Actions.pdf')
    
    f =  open('test.txt', "w")
    f.write(new_data)
    f.close()
        
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December' ]
    firstMonth = 'month'
    lowestIndex = 1000000
    monthIndex = -1
    
    for month in months:
        newIndex = new_data.find(month)
        if (newIndex != -1 and lowestIndex > newIndex):
            lowestIndex = newIndex
            firstMonth = month
            monthIndex = months.index(month)

    currentMonthIndex = (monthIndex + 2) % 12
    
    new_data_array_regex = standardre.compile('\(FINRA\n*\s\n*Case\n*\s\n*#\d+\)').split(new_data)

    
    new_data_array = ['The', 'the']
    for t in new_data_array_regex:
        t = t.replace('\n', ' ')
        
        t = t.replace('Individuals', '')
        t = t.replace('Sanctioned', '')
        t = t.replace('CRD', '')
        t = t.replace('Firms Fined', '')
        t = t.replace('Suspended', '')
        t = t.replace('Individuals', '')
        t = t.replace('Barred', '')
        t = t.replace('FINRA', '')
        t = t.replace('  and  ', '')
        t = t.replace('Actions', '')
        t = t.replace('Other', '')
        t = t.replace('Filed', '')
        t = t.replace('Suspended', '')
        t = t.replace('Complaints', '')
        t = t.replace('Decision', '')
        t = t.replace('Disciplinary', '')   
        t = t.replace('\x0c', '')
        t = t.replace('2018', '')
        
        
        t = t.replace(months[currentMonthIndex], ' ')
        t = standardre.sub('[\s+]', ' ', t)
        t = t.strip()
        t = standardre.sub('[\d+]', '', t)
        

        s = t.split(firstMonth)
        print(len(s))
        print(s[0])
        new_data_array.append(s[0])
     
    
    entities = []    
    
    
    # Run multiple times in order to improve accuracy. 
    for new_text in new_data_array:
        doc = nlp(new_text)        

    for new_text in new_data_array:
        doc = nlp(new_text)
        for ent in doc.ents:
           if (ent.text, ent.label_) not in entities:
                entities.extend([(ent.text, ent.label_)])
    
    doc = nlp("Random string")
    
    for new_text in new_data_array:
        print(new_text)
        doc = nlp(new_text)
        for ent in doc.ents:
            if (ent.text, ent.label_) not in entities:
                entities.extend([(ent.text, ent.label_)])
        
        # entities.extend([(ent.text, ent.label_) for ent in doc.ents])
        # print([(ent.text, ent.label_) for ent in doc.ents])
        
    print('.')
    print('.')
    print('.')
    print('.')
    print('.')
    print('.')

    print(entities);
    
   # doc = nlp(new_data)
   # print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
   # print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])


    file = open("FINRA_entities.txt", "a")
    for entity in entities :
        file.write("\n")
        file.write(str(entity))
    file.close()

   # doc = nlp(new_data)
   # print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
   # print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
            print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

if __name__ == '__main__':
    plac.call(main)