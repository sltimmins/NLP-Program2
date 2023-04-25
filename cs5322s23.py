import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import re

rubbish_model = 'rubbish-word-sense'
tissue_model = 'tissue-word-sense'
yarn_model = 'yarn-word-sense'

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def remove_quotes(line):
    if line.startswith('"'):
        line = line[1:]
    if line.endswith('"'):
        line = line[:-1]
    return line

def preprocess_sentence(s, word, senses):
    s += f' [SEP] {word}'
    for sense in senses:
        s += f' [SEP] {sense}'
    return s

def parse_file_to_df(filename):
    with open(filename) as f:
        lines = [remove_quotes(line.strip()) for line in f.readlines()]
        word = lines[0]
        senses = []
        
        i = 2
        for i in range(2, len(lines)):
            if not re.search(r'^[0-9]:? \([a-z]+\)', lines[i]):
                break
            else:
                senses.append(lines[i])
        
        curr_sense = 1
        sentences = []
        sense = []
        for i in range(i, len(lines)):
            if not lines[i]:
                continue
            if re.match(r'[0-9]', lines[i]):
                curr_sense = int(lines[i])
            else:
                s = lines[i].strip()
                sentences.append(preprocess_sentence(s, word ,senses))
                sense.append(curr_sense - 1)
            
        
        
        return senses, pd.DataFrame({"sentence": sentences, "sense": sense})
    
rubbish = parse_file_to_df('rubbish.txt')
tissue = parse_file_to_df('tissue.txt')
yarn = parse_file_to_df('yarn.txt')
    
def WSD_test_rubbish(sentences):
    pipe = pipeline("text-classification", model=rubbish_model, tokenizer=tokenizer)
    predictions = []
    for sentence in sentences:
        pred = pipe(preprocess_sentence(sentence, 'rubbish', rubbish[0]))
        if pred[0]['label'] == 'LABEL_0': predictions.append(1)
        else: predictions.append(2)
        
    with open('result_rubbish_sam_timmins.txt', 'w') as f:
        for p in predictions:
            f.write(f'{p}\n')
            
def WSD_test_tissue(sentences):
    pipe = pipeline("text-classification", model=tissue_model, tokenizer=tokenizer)
    predictions = []
    for sentence in sentences:
        pred = pipe(preprocess_sentence(sentence, 'tissue', tissue[0]))
        if pred[0]['label'] == 'LABEL_0': predictions.append(1)
        else: predictions.append(2)
        
    with open('result_tissue_sam_timmins.txt', 'w') as f:
        for p in predictions:
            f.write(f'{p}\n')
            
def WSD_test_yarn(sentences):
    pipe = pipeline("text-classification", model=yarn_model, tokenizer=tokenizer)
    predictions = []
    for sentence in sentences:
        pred = pipe(preprocess_sentence(sentence, 'yarn', yarn[0]))
        if pred[0]['label'] == 'LABEL_0': predictions.append(1)
        else: predictions.append(2)
        
    with open('result_yarn_sam_timmins.txt', 'w') as f:
        for p in predictions:
            f.write(f'{p}\n')
            
with open('rubbish_test.txt') as f:
    sentences = [remove_quotes(line.strip()) for line in f.readlines()]
    WSD_test_rubbish(sentences)
    
with open('tissue_test.txt') as f:
    sentences = [remove_quotes(line.strip()) for line in f.readlines()]
    WSD_test_tissue(sentences)
    
with open('yarn_test.txt') as f:
    sentences = [remove_quotes(line.strip()) for line in f.readlines()]
    WSD_test_yarn(sentences)