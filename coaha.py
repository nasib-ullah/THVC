'''
Module: coaha 
Description: script to compute COAHA metric.
Author: nasibullah104@gmail.com
paper: Thinking Hallucination for Video Captioning.
'''

import os
from tqdm import tqdm
import gensim.downloader
import torch
import torch.nn as nn
import random
import numpy as np
try:
    import pickle5 as pickle
except:
    import pickle

import spacy 
import nltk
from collections import Counter
import nltk

class COAHA:
    
    def __init__(self,cfg,gt_dict):
        
        self.sno = nltk.stem.SnowballStemmer('english')
        
        #Common stopwords to handle noisy POS and NER of spacy.
        self.frequent_words_object = ['a','is','a','s','are','two','up','the','some','with','his','her']
        self.frequent_words_action = ['a','is','a','face','s','are','two','up','rope','the','car','bowl','flour','ball',
                                'bread','face','with','his','her','dog','bike']
        
        self.action_add = []
        f = open('action_list','r')
        for line in f.readlines():
            self.action_add.append(self.sno.stem(line.strip()))
        self.object_dict = {}
        self.action_dict = {}
        self.object_list = []
        self.action_list = []
        self.coaha_list = []
        self.coaha_dict = {}
        self.oh = {}
        self.ah = {}
        self.coaha_total = 0
        if cfg.semantic_embedder=='glove':
            self.embedder = gensim.downloader.load('glove-wiki-gigaword-300')
        elif cfg.semantic_embedder=='fasttext':
            self.embedder = gensim.downloader.load('fasttext-wiki-news-subwords-300')
        elif cfg.semantic_embedder=='combine':
            self.glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')
        else:
            print('Choose proper Semantic Embedder')
        self.gt_dict = gt_dict
        self.object_dict,self.action_dict = self._Entity_Extraction_COAHA()
        self.object_list = [str(x) for x in self.object_list if str(x) not in self.frequent_words_object]
        
        self.action_list = [str(x) for x in self.action_list if str(x) not in self.frequent_words_action]
        self.action_list = [str(x) for x in self.action_list if str(x) not in self.object_list] + self.action_add
        self.object_list = [str(x) for x in self.object_list if str(x) not in self.action_list]
        
        self.oov = 0
        self.hallucinated_object_dict = {}
        self.hallucinated_action_dict = {}
        self.predicted_object_list = []
        self.predicted_action_list = []
        self.avg_sentence_length()
        
    def avg_sentence_length(self):
        self.avg_length = {}
        for k,v in self.gt_dict.items():
            length = 0
            i = 0 
            for sentence in v:
                length += len(sentence.split(' '))
                i += 1
            self.avg_length[k] = length/i 
            
        
    def predicted_object_action_detection(self,prediction_dict):
        nlp = spacy.load("en_core_web_trf")
        object_list = []
        action_list = []
        
        for k,v in tqdm(self.gt_dict.items()):
            parse_texts = nlp(v[0])
        
            for parse_text in parse_texts:
                if parse_text.pos_ == 'VERB':
                    action_list.append(self.sno.stem(str(parse_text)))
                    #print(parse_text,parse_text.pos_)
                if parse_text.dep_ == 'nsubj':
                    object_list.append(self.sno.stem(str(parse_text)))
                    #print(text,text.dep_,text.orth_)
                if parse_text.dep_ == 'iobj':
                    pass
                    #print(text,text.dep_)
                if parse_text.dep_ == 'dobj':
                    object_list.append(self.sno.stem(str(parse_text)))
                    #print(text,text.dep_)
         
        object_list = list(set(object_list))
        action_list = list(set(action_list))
        
        return object_list, action_list 

        
    def _Entity_Extraction_COAHA(self):
        
        nlp = spacy.load("en_core_web_trf") 
        sno = nltk.stem.SnowballStemmer('english')

        object_dict = {}
        action_dict = {}
        Yact = []
        Yobj = []
        print('Setting up COAHA object...')
        for k,v in tqdm(self.gt_dict.items()):
            text = str(' '.join([x for x in self.gt_dict[k]]))
            #print(text)
            parse_texts = nlp(text)
            N_O = []
            N_A = []
            for parse_text in parse_texts:
                if parse_text.pos_ == 'VERB':
                    N_A.append(sno.stem(str(parse_text)))
                    #print(parse_text,parse_text.pos_)
                if parse_text.dep_ == 'nsubj':
                    N_O.append(sno.stem(str(parse_text)))
                    #print(text,text.dep_,text.orth_)
                if parse_text.dep_ == 'iobj':
                    pass
                    #print(text,text.dep_)
                if parse_text.dep_ == 'dobj':
                    N_O.append(sno.stem(str(parse_text)))
                    #print(text,text.dep_)

            obj_c = list(set(N_O)) # take all
            act_c = list(set(N_A)) #take all


            object_dict[k] = obj_c
            action_dict[k] = act_c

            Yobj += obj_c #take all objects
            Yact += act_c #take all actions


        self.action_list = list(set(Yact))
        self.object_list = list(set(Yobj))

        return object_dict,action_dict
    
    def evaluate(self,prediction_dict):
        self.predicted_object_list, self.predicted_action_list = self.predicted_object_action_detection(prediction_dict)
        self.predicted_object_list = [str(x) for x in self.predicted_object_list if str(x) not in self.frequent_words_object]
        self.predicted_object_list = [str(x) for x in self.predicted_object_list if str(x) not in self.action_list]
        self.predicted_action_list = [str(x) for x in self.predicted_action_list if str(x) not in self.frequent_words_action]
        self.predicted_action_list = [str(x) for x in self.predicted_action_list if str(x) not in self.object_list]
        
        for k,v in prediction_dict.items():
            self.oh[k],self.ah[k] = self.evaluate_single(v[0],k)
            try:
                coaha = self.oh[k] + self.ah[k]
            except:
                coaha=0
            self.coaha_dict[k] = coaha
            self.coaha_list.append(coaha) 
        self.coaha_total = sum(self.coaha_list)/len(self.coaha_list)
        
    def evaluate_single(self,predicted,key):
        oh = 0
        ah = 0
        words = predicted.split(' ')
        words = [self.sno.stem(word) for word in words]
        hallucinated_objects = [word for word in words if word in self.object_list ]+ [word for word in words if word in self.predicted_object_list ]
        hallucinated_objects = [word for word in hallucinated_objects if word not in self.object_dict[key] ]
        hallucinated_objects = list(set(hallucinated_objects))
        
        hallucinated_actions = [word for word in words if word in self.action_list ]+  [word for word in words if word in self.predicted_action_list ]
        hallucinated_actions = [word for word in hallucinated_actions if word not in self.action_dict[key] ]
        hallucinated_actions = list(set(hallucinated_actions))
        
        
        self.hallucinated_object_dict[key] = hallucinated_objects
        self.hallucinated_action_dict[key] = hallucinated_actions
        
        for word in hallucinated_objects:
            oh += self._distance_calculation(word,self.object_dict[key])
        for word in hallucinated_actions:    
            ah += self._distance_calculation(word,self.action_dict[key])
        oh /= self.avg_length[key]     
        ah /= self.avg_length[key]     
        return oh,ah
            
    def _distance_calculation(self,word,lst):
        distance = 0
        for wrd in lst:
            try:
                distance += self.embedder.distance(word,str(wrd))
            except:
                self.oov += 1
        distance = distance /len(lst)
        return distance
               

