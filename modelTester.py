#!/usr/bin/env python
# coding: utf-8

#Package imports 
import pandas as pd 
from transformers import BertTokenizer, DataCollatorWithPadding, BertForMaskedLM
import torch.nn #Parent module for pytorch for neural networks 
import torch.nn.functional as F #for the activation function 
from torch.utils.data import DataLoader
import sys
from datasets import Dataset
from nltk.corpus import wordnet
from torch.nn.functional import softmax 

#Find test file: supports input from user on CLI
def scanFile(): 
    if sys.argv[1] != '':
        testFile = str(sys.argv[1])
        
    else: 
        testFile = 'testSet.csv'

        #Append .csv if necessary (Probably not necessary)
        if testFile[-4:] != '.csv':
            testFile = testFile + '.csv'
    return "./" + testFile 

#Initialize ML components 
model = BertForMaskedLM.from_pretrained('Brijhs/Saama-model') #Personal model from HF_hub
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')
dataCollator = DataCollatorWithPadding(tokenizer)

#Helper functions for masking and tokenizing 
def replaceUnderscore(word): 
    strList = [*word]
    for i in range(len(strList)): 
        if strList[i] == '_': 
            strList[i] = '[MASK]'
    
    finalStr = ""
    for i in range(len(strList)): 
        finalStr += strList[i]
    return finalStr

def spaceWord(word): 
    if type(word) != word: 
        word = str(word)

    k = len(word)
    wordList = [*word]
    newList = []
    
    for i in range(k): 
        newList.append(wordList[i])
        newList.append(' ')

    return ''.join(newList)

def getFirstDef(word): 
    return wordnet.synsets(word)[0].definition()

def tokenize_maskOnly(example):
    return tokenizer(example['masked'], padding = "max_length", max_length = 128, truncation = True)

def checkPrediction(prediction, discarded):
    final = []
    for i in range(len(prediction)): 
        if (i + 1)%64 == 0: 
            print("Prediction counter: {}".format(str(i + 1)))
        current = tokenizer.decode(prediction[i])
        for element in current: 
            if element not in discarded: 
                final.append(element)
    return final 

def interpretPrediction(outputLogits): 
    discarded = ['[', ']', ' ', 'P', 'A', 'D']
    sm = softmax(outputLogits, dim = -1)

    prediction = torch.argmax(sm, dim= -1)
    final = []
    for i in range(len(prediction)): 
        for item in prediction[i]: 
            if item != 0: 
                final.append(item)

    return checkPrediction(final, discarded)

#Takes test file as pandas series of the given configuration and returns DataLoader with input_ids and attention_mask 
def createLoader(testSet): 
    maskedDict = {'masked': []}
    for i in range(len(testSet)):
        maskedDict['masked'].append("{} [SEP] ".format(testSet['Meaning'][i]) + replaceUnderscore(testSet['Masked'][i]))
    
    m_ds = Dataset.from_dict(maskedDict)
    mapped = m_ds.map(tokenize_maskOnly) 
    mapped = mapped.remove_columns(['token_type_ids', 'masked'])
    
    tokenized = mapped.with_format("torch")
    loader = DataLoader(tokenized, batch_size = 1, collate_fn = dataCollator)
    return loader


#Main run screipt 
if __name__ == "__main__": 
    testFile = scanFile()
    test = pd.read_csv(testFile)    
    modelResponse = {'Results': []}
    loader = createLoader(test)

    #Caculation loop 
    bc = 0 
    for batch in loader:
        
        #For updating user watching computation 
        bc += 1
        if bc%50 == 0: 
            print("Batch number: {}".format(str(bc))) 

        input = batch['input_ids']
        attentionMask = batch['attention_mask']
        output = (model(input, attention_mask = attentionMask))
        result = interpretPrediction(output.logits)
        modelResponse['Results'].append(result)

    #Have to load in data and batch, then feed in loop to model 

    test['Result'] = modelResponse['Results']
    print(test)