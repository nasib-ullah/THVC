'''
Module :  data
Details : This module creates datasets and dataloaders suitable for feeding data to models.
          It Currently supports MSVD and MSRVTT.
          
'''

import os
import random
import json
import h5py
import itertools
from PIL import Image
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from torch.nn import functional as F
from tqdm import tqdm
try:
    import pickle5 as pickle
except:
    import pickle

import spacy 
import nltk
from collections import Counter



def collate_fn(batch): # add support for motion and object features
    '''
    Custom collate function for supporting batching during training and inference. 
    '''
   
    data=[item[0] for item in batch]
    images=torch.stack(data,0)    
    label=[item[1] for item in batch]
    ides = [item[2] for item in batch]
    
    motion = [item[3] for item in batch]
    motion_batch = torch.stack(motion,0)
    
    object_ = [item[4] for item in batch]
    object_batch = torch.stack(object_,0)
    
    aux_obj = [item[5] for item in batch]
    aux_obj = torch.stack(aux_obj,0)
    
    aux_act = [item[6] for item in batch]
    aux_act = torch.stack(aux_act,0)
    

    max_target_len = max([len(indexes) for indexes in label])
    padList = list(itertools.zip_longest(*label, fillvalue = 0))

    lengths = torch.tensor([len(p) for p in label])
    padVar = torch.LongTensor(padList)

    m = []
    for i, seq in enumerate(padVar):
        #m.append([])
        tmp = []
        for token in seq:
            if token == 0:
                tmp.append(int(0))
            else:
                tmp.append(1)
        m.append(tmp)
    m = torch.tensor(m)
    
    return images,padVar,m,max_target_len,ides,motion_batch,object_batch,aux_obj,aux_act
        
class CustomDataset(Dataset):
    
    def __init__(self,cfg,appearance_feature_dict, annotation_dict , video_name_list,voc,motion_feature_dict,
                     object_feature_dict,auxhead_data=None):
        
        self.annotation_dict = annotation_dict
        self.appearance_feature_dict = appearance_feature_dict
        self.v_name_list = video_name_list
        self.voc = voc
        self.max_caption_length = cfg.max_caption_length
        self.motion_feature_dict = motion_feature_dict
        self.object_feature_dict = object_feature_dict 
        self.opt_truncate_caption = cfg.opt_truncate_caption
        self.opt_auxhead = cfg.opt_auxiliary_heads
        self.auxhead_data = auxhead_data
        
    def __len__(self):
        return len(self.v_name_list)
    
    def __getitem__(self,idx): 
        
        anno = random.choice(self.annotation_dict[self.v_name_list[idx]])
        anno_index = []
        if self.auxhead_data is not None and self.opt_auxhead:
            object_gt = torch.zeros(len(self.auxhead_data['object_list']))
            action_gt = torch.zeros(len(self.auxhead_data['action_list']))
        else:
            object_gt = torch.zeros(100)
            action_gt = torch.zeros(100)
            
        for word in anno.split(' '):
            try:
                anno_index.append(self.voc.word2index[word])
            except:
                pass 
            
            if self.auxhead_data is not None and self.opt_auxhead:
                if word in self.auxhead_data['object_list']:
                    object_gt[self.auxhead_data['object_list'].index(word)] = 1
                if word in self.auxhead_data['action_list']:
                    action_gt[self.auxhead_data['action_list'].index(word)] = 1
        if self.opt_truncate_caption:
            if len(anno_index)> self.max_caption_length:
                anno_index = anno_index[:self.max_caption_length]
        anno_index = anno_index + [self.voc.cfg.EOS_token]
        
        appearance_tensor = torch.tensor(self.appearance_feature_dict[self.v_name_list[idx]]).float()
        
        if self.motion_feature_dict == None:
            motion_tensor = torch.zeros_like(appearance_tensor)
        else:
             motion_tensor = torch.tensor(self.motion_feature_dict[self.v_name_list[idx]]).float()
        if self.object_feature_dict == None:
            object_tensor = torch.zeros_like(appearance_tensor)
        else:
            object_tensor = torch.tensor(self.object_feature_dict[self.v_name_list[idx]]).float()
            
        return appearance_tensor,anno_index, self.v_name_list[idx],motion_tensor,object_tensor,object_gt,action_gt
            

class DataHandler:
    
    def __init__(self,cfg,path,voc):
        
        self.voc = voc
        self.cfg = cfg
        self.path = path
        self.appearance_feature_dict = {}
        self.motion_feature_dict = {}
        self.object_feature_dict = {}
        self.object_dict = {}
        self.auxhead_data = {}         

        if cfg.dataset == 'msvd': 
            self.vid2url = self._name_mapping(path.name_mapping_file)   
            self._msvd_create_dict() # Reference caption dictionaries
            #read appearance feature file
            self.appearance_feature_dict = self._read_feature_file(feature_type='appearance')
            #read motion and object feature file
            self.motion_feature_dict = self._read_feature_file(feature_type='motion')
            self.object_feature_dict = self._read_feature_file(feature_type='object')
            if cfg.opt_auxiliary_heads:
                if cfg.create_entity:
                    self.Entity_Extraction()
                    self.Save_AuxHead_data()
                else:
                    path = os.path.join('Saved','auxhead_data_MSVD.p')
                    with open(path, 'rb') as fp:
                        self.auxhead_data = pickle.load(fp)
                
        if cfg.dataset == 'msrvtt':
            self.train_dict, self.val_dict,self.test_dict = self._msrvtt_create_dict() # Reference caption dictionaries
            # read appearance feature file
            self.appearance_feature_dict = self._read_feature_file(feature_type='appearance')
            #read motion and object feature file
            self.motion_feature_dict = self._read_feature_file(feature_type='motion')
            self.object_feature_dict = self._read_feature_file(feature_type='object')
            if cfg.opt_auxiliary_heads:
                if cfg.create_entity:
                    self.Entity_Extraction()
                    self.Save_AuxHead_data()
                else:
                    path = os.path.join('Saved','auxhead_data_MSRVTT.p')
                    with open(path, 'rb') as fp:
                        self.auxhead_data = pickle.load(fp)
                
        self.train_name_list = list(self.train_dict.keys())
        self.val_name_list = list(self.val_dict.keys())
        self.test_name_list = list(self.test_dict.keys())
        
    def _read_feature_file(self,feature_type='appearance'):
        
        feature_dict = {}
        if feature_type == 'appearance':
            f1 = h5py.File(self.path.appearance_feature_file,'r+')
        elif feature_type == 'motion':
            f1 = h5py.File(self.path.motion_feature_file,'r+')
        else:
            f1 = h5py.File(self.path.object_feature_file,'r+')
            
        for key in f1.keys():
            arr = f1[key].value
            if arr.shape[0] < self.cfg.frame_len:
                pad = self.cfg.frame_len - arr.shape[0]
                arr = np.concatenate((arr,np.zeros((pad,arr.shape[1]))),axis = 0)
            if arr.shape[0] > self.cfg.frame_len:
                arr = arr[:self.cfg.frame_len]
            feature_dict[key] = arr
                
        return feature_dict

    def _file_to_dict(self,path):
        dic = dict()
        fil = open(path,'r+')
        for f in fil.readlines():
            l = f.split() 
            ll = ' '.join(x for x in l[1:])
            if l[0] not in dic:
                dic[l[0]] = [ll]
            else:
                dic[l[0]].append(ll)
        return dic
    
    def _name_mapping(self,filename):
        vid2url = dict()
        fil = open(filename,'r+')
        for f in fil.readlines():
            l = f.split(' ')
            vid2url[l[1].strip('\n')] = l[0]
        return vid2url
    
    def _msvd_create_dict(self):
        self.train_dict = self._file_to_dict(self.path.train_annotation_file)
        self.val_dict = self._file_to_dict(self.path.val_annotation_file)
        self.test_dict = self._file_to_dict(self.path.test_annotation_file)
            
        
    def _msrvtt_create_dict(self):
        train_val_file = json.load(open(self.path.train_val_annotation_file))
        test_file = json.load(open(self.path.test_annotation_file))
        train_dict = {}
        val_dict = {}
        test_dict = {}
        for datap in train_val_file['sentences']:
            if int(datap['video_id'][5:]) in self.path.train_id_list:
                if datap['video_id'] in list(train_dict.keys()):
                    train_dict[datap['video_id']] += [datap['caption']]
                else:
                    train_dict[datap['video_id']] = [datap['caption']]
            if int(datap['video_id'][5:]) in self.path.val_id_list:
                if datap['video_id'] in list(val_dict.keys()):
                    val_dict[datap['video_id']] += [datap['caption']]
                else:
                    val_dict[datap['video_id']] = [datap['caption']]
            
        for datap in test_file['sentences']:
            if datap['video_id'] in list(test_dict.keys()):
                test_dict[datap['video_id']] += [datap['caption']]
            else:
                test_dict[datap['video_id']] = [datap['caption']]
        return train_dict,val_dict,test_dict
    
    def getDatasets(self):
        
        train_dset = CustomDataset(self.cfg,self.appearance_feature_dict, self.train_dict,self.train_name_list, self.voc,
                                  self.motion_feature_dict,self.object_feature_dict,self.auxhead_data)
        
        val_dset = CustomDataset(self.cfg,self.appearance_feature_dict, self.val_dict, self.val_name_list, self.voc,
                                self.motion_feature_dict,self.object_feature_dict,self.auxhead_data)
        
        test_dset = CustomDataset(self.cfg,self.appearance_feature_dict, self.test_dict, self.test_name_list, self.voc,
                                 self.motion_feature_dict,self.object_feature_dict,self.auxhead_data)
                
        return train_dset, val_dset, test_dset
    
    def getDataloader(self,train_dset,val_dset,test_dset):
        
        train_loader=DataLoader(train_dset,batch_size = self.cfg.batch_size, num_workers = 8,shuffle = True,
                        collate_fn = collate_fn, drop_last=True)

        val_loader = DataLoader(val_dset,batch_size = self.cfg.val_batch_size, num_workers = 8,
                                shuffle = False,collate_fn = collate_fn,drop_last=False)
        test_loader = DataLoader(test_dset,batch_size = self.cfg.val_batch_size, num_workers = 8,shuffle = False,
                                 collate_fn = collate_fn,drop_last=False)
        
        return train_loader,val_loader,test_loader
    
    def Save_AuxHead_data(self):
        if self.cfg.dataset == 'msvd':
            path = os.path.join('Saved','auxhead_data_MSVD.p')
        else:
            path = os.path.join('Saved','auxhead_data_MSRVTT.p')
        with open(path, 'wb') as fp:
            pickle.dump(self.auxhead_data , fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
    def Entity_Extraction(self):
        
        print('Extracting entities for auxiliary heads...')
        nlp = spacy.load("en_core_web_trf") 
        sno = nltk.stem.SnowballStemmer('english')

        object_dict = {}
        action_dict = {}
        Yact = []
        Yobj = []
        for k,v in tqdm(self.train_dict.items()):
            text = str(' '.join([x for x in self.train_dict[k]]))
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

            dict_obj = Counter(N_O) #Top 1
            dict_act = Counter(N_A) # Top 1
            dict_obj = {v: k for k, v in dict_obj.items()} # top 1
            dict_act = {v: k for k, v in dict_act.items()} #top 1

            obj_c = dict_obj[max(dict_obj.keys())] #top 1
            act_c = dict_act[max(dict_act.keys())] # top 1

            #obj_c = list(set(N_O)) # take all
            #act_c = list(set(N_A)) #take all


            object_dict[k] = obj_c
            action_dict[k] = act_c

            #Yobj += obj_c #take all objects
            #Yact += act_c #take all actions
            Yobj.append(obj_c) #top 1
            Yact.append(act_c) #top 1

        Yact = list(set(Yact))
        Yobj = list(set(Yobj))
        
        #Yact = Yact[:500]
        #Yobj = Yobj[:500]
        
        self.auxhead_data['action_dict'] = action_dict
        self.auxhead_data['object_dict'] = object_dict
        self.auxhead_data['action_list'] = Yact
        self.auxhead_data['object_list'] = Yobj
