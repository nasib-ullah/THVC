'''
Module : evaluate
'''

import os
import sys
import torch
import json
import pickle

class Evaluator:
    
    def __init__(self,model,dataloader,path,cfg,reference_dict,decoding_type = 'greedy'):
        '''
        Decoding type : {'greedy','beam'}
        '''
        self.path = path
        self.cfg = cfg
        self.dataloader = dataloader
        self.reference_dict = reference_dict
        self.prediction_dict = {}
        self.scores = {}
        self.losses = {}
        self.best_model = model
        self.decoding_type = decoding_type

    def prediction_list(self,model):
        self.prediction_dict = {}
        ide_list = []
        caption_list = []
        model.eval()
        with torch.no_grad():
            for data in self.dataloader:
                features, targets, mask, max_length,ides,motion_feat,object_feat,_,_= data
                if self.decoding_type == 'greedy':
                    cap,cap_txt,_,_ = model.GreedyDecoding(features.to(self.cfg.device),motion_feat.to(self.cfg.device),
                                                              object_feat.to(self.cfg.device))
                else:
                    cap_txt = model.BeamDecoding(features.to(self.cfg.device),
                                                     motion_feat.to(self.cfg.device),
                                                     object_feat.to(self.cfg.device),self.cfg.beam_length)
                    
                ide_list += ides
                caption_list += cap_txt
        for a in zip(ide_list,caption_list):
            self.prediction_dict[str(a[0])] = [a[1].strip()]
            
    def evaluate(self,scorer,model,epoch,loss=9999):
        self.prediction_list(model)
        scores = scorer.score(self.reference_dict,self.prediction_dict)
        self.scores[epoch] = scores
        self.losses[epoch] = loss
        return scores


    def save_model(self,model,epoch):
        print('Saving models....')
        filename = os.path.join(self.path.saved_models_path, self.cfg.model_name+str(epoch)+'.pt')
        torch.save(model,filename)
