
'''
Module :  THVC model
Details : Implementation of the paper Thinking Hallucination for Video Captioning.
          
Notations : B : Batch_size, T : Frame dimension, F : dimension of pre-trained CNN extracted features,
            F' : projected feature dimension, E : Word embedding, h : decoder hidden size.

'''

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import torchvision
import torchvision.transforms as transforms

import random
import itertools
import math
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import itertools

import numpy as np
import os
import copy


class Encoder(nn.Module):
    
    def __init__(self,cfg):
        super(Encoder,self).__init__()
        '''
        Encoder module. Project the video appearance and motion features into a different space which will be 
        send to the decoder.
        Argumets:
          input_size : CNN extracted feature size. [resnet101,Resnext101 - 2048, Inceptionv4,Inceptionresnetv2 - 1536]
          output_size : Dimention of projected space.
        '''
        
        self.appearance_projection_layer = nn.Linear(cfg.appearance_input_size,cfg.appearance_projected_size)
        self.motion_projection_layer = nn.Linear(cfg.motion_input_size ,cfg.motion_projected_size)
        self.object_projection_layer = nn.Linear(cfg.object_input_size ,cfg.object_projected_size)
        
    def forward(self,appearance_feat,motion_feat,object_feat):
        '''
        Project 2D and 3D convnet extracted features.
        Args:
            appearance_feat : (B,T,F)
            motion_feat : (B,T,F)
        Return:
            appearance_out : (B,T,F')
            motion_out : (B,T,F')
        
        '''
        appearance_out = self.appearance_projection_layer(appearance_feat)
        motion_out = self.motion_projection_layer(motion_feat)
        object_out = self.object_projection_layer(object_feat)
        
        return appearance_out, motion_out, object_out
    
class Auxiliary_heads(nn.Module):
    
    def __init__(self,cfg):
        super(Auxiliary_heads,self).__init__()
        
        self.object_head = nn.Linear(cfg.object_projected_size,cfg.object_head_number)
        self.action_head = nn.Linear(cfg.motion_projected_size,cfg.action_head_number)
        
    def forward(self,motion_feature,object_feature):
        '''
        motion_feature: (B,28,512)
        object_feature: (B,28,512)
        
        '''
        object_out = self.object_head(object_feature) #(B,28,195)
        action_out = self.action_head(motion_feature) #(B,28,223)
        
        return object_out.mean(dim=1),action_out.mean(dim=1) #(B,195) (B,223)
    
    
class TemporalAttention(nn.Module):
    def __init__(self,cfg):
        super(TemporalAttention,self).__init__()
        '''
        Temporal Attention module. It depends on previous hidden memory in the decoder,
        feature at the source side 
        at(s) = align(ht,hs)
              = exp(score(ht,hs)) / Sum(exp(score(ht,hs')))  
        where
        score(ht,hs) = ht.t * hs                         (dot)
                     = ht.t * Wa * hs                  (general)
                     = va.t * tanh(Wa[ht;hs])           (concat)  
        Here we have used concat formulae.
        Argumets:
          hidden_size : hidden memory size of decoder. (batch,hidden_size)
          feat_size : feature size of each frame at encoder side.
          bottleneck_size : intermediate size.
        '''
        
        
        self.hidden_size = cfg.decoder_hidden_size
        self.feat_size = cfg.feat_size
        self.bottleneck_size = cfg.attn_size
        
        self.decoder_projection = nn.Linear(self.hidden_size,self.bottleneck_size)
        self.encoder_projection = nn.Linear(self.feat_size, self.bottleneck_size)
        self.final_projection = nn.Linear(self.bottleneck_size,1)
        
    def forward(self,hidden,feats):
        '''
        shape of hidden : (hidden_size) (batch,hidden_size) #(100,512)
        shape of feats : (batch size,feat_size)  #(100,40,1536)
        '''
        Wh = self.decoder_projection(hidden)  
        Uv = self.encoder_projection(feats)   
        Wh = Wh.unsqueeze(1).expand_as(Uv)
        energies = self.final_projection(torch.tanh(Wh+Uv))
        weights = F.softmax(energies, dim=1)
        weighted_feats = feats *weights.expand_as(feats)
        attn_feats = weighted_feats.sum(dim=1)
        return attn_feats,weights

    
class EncoderContextGate(nn.Module):
    
    def __init__(self,cfg):
        super(EncoderContextGate,self).__init__()
        self.split_shape = cfg.appearance_projected_size
        self.bottleneck = cfg.EncoderCG_bottleneck_size
        self.context_projection_layer = nn.Linear(cfg.projected_feature_size,self.bottleneck)
        self.visual_projection_layer = nn.Linear(cfg.projected_feature_size,self.bottleneck)
        self.language_projection_layer = nn.Linear(cfg.embedding_size,self.bottleneck)
        
        self.out_layer = nn.Linear(self.bottleneck,cfg.projected_feature_size)
        
    def forward(self,visual_memory,language_memory,context_vector):
        '''
        visual_memory: (B,1536)
        language_memory: (B,E)
        context_vector: (1,B,1536)
        decoder_hidden: (1,B,512)
        
        '''
        cv = self.context_projection_layer(context_vector.squeeze(0)) # (100,512) 
        vv = self.visual_projection_layer(visual_memory) # (100,512)
        lv = self.language_projection_layer(language_memory) # (100,512)
        out = F.sigmoid(self.out_layer(cv+vv+lv)) # (100,512)
        CGA,CGM,CGO = out.split(self.split_shape,dim=1)
        return CGA,CGM,CGO
    
    
class DecoderContextGate(nn.Module):
    
    def __init__(self,cfg):
        super(DecoderContextGate,self).__init__()
        self.bottleneck = cfg.DecoderCG_bottleneck_size
        
        self.visual_projection_layer = nn.Linear(cfg.projected_feature_size,self.bottleneck)
        self.language_projection_layer = nn.Linear(cfg.embedding_size,self.bottleneck)

        
        self.out_source = nn.Linear(self.bottleneck,cfg.projected_feature_size)
        self.out_target = nn.Linear(self.bottleneck,cfg.embedding_size)
        
    def forward(self,visual_memory,language_memory,context_vector,target_context):
        '''
        visual_memory: (B,1536)
        language_memory: (B,E)
        context_vector: (1,B,1536)
        target_context: (1,B,E)
        
        '''
        
        vc = visual_memory + context_vector.squeeze(0) #(B,2*1536)
        vc = self.visual_projection_layer(vc)
        tc = language_memory + target_context.squeeze(0) # (B,2*E)
        tc = self.language_projection_layer(tc)
        SG = F.sigmoid(self.out_source(vc+tc)) # (100,512)
        TG = F.sigmoid(self.out_target(vc+tc))
        
        return SG.unsqueeze(0),TG.unsqueeze(0)
        
    
class RecurrentDecoder(nn.Module):
    
    def __init__(self,cfg,voc):
        super(RecurrentDecoder, self).__init__()
        '''
        Attention-based Recurrent Decoder
        Args:
            cfg : Configuration object corresponding to MARN
            voc : Vocabulary object
        
        '''
        
        # Keep for reference
        self.opt_EncoderCG = cfg.opt_EncoderCG
        self.opt_DecoderCG = cfg.opt_DecoderCG
        self.dropout = cfg.dropout
        self.feat_len = cfg.frame_len
        self.attn_size = cfg.attn_size
        self.output_size = voc.num_words
        self.rnn_dropout = cfg.rnn_dropout
        self.n_layers = cfg.n_layers
        self.decoder_type = cfg.decoder_type

        # Define layers
        self.embedding = nn.Embedding(voc.num_words, cfg.embedding_size)
        self.attention = TemporalAttention(cfg)
        self.embedding_dropout = nn.Dropout(cfg.dropout)
        
        if cfg.opt_EncoderCG:
            self.EncoderCG = EncoderContextGate(cfg)
            
        if cfg.opt_DecoderCG:
            self.DecoderCG = DecoderContextGate(cfg)
        
        
        if self.decoder_type == 'gru':
            self.rnn = nn.GRU(input_size=cfg.decoder_input_size, hidden_size=cfg.decoder_hidden_size,
                              num_layers=self.n_layers, dropout=self.rnn_dropout)
        else:
            self.rnn = nn.LSTM(input_size=cfg.decoder_input_size, hidden_size=cfg.decoder_hidden_size,
                           num_layers=self.n_layers, dropout=self.rnn_dropout)
        self.out = nn.Linear(cfg.decoder_hidden_size, self.output_size)

    
    def forward(self, inputs, hidden, appearance_feats,motion_feats,object_feats,vis_mem,lang_mem):
        '''
        we run this one step (word) at a time
        
        inputs -  (1, B)
        hidden - h_n/c_n :(num_layers * num_directions, B, hidden_size)    # GRU:h_n   LSTM:(h_n,c_n)
        appearance_feats - (B,T,F')
        motion_feats : (B,T,F')
        
        '''
        embedded = self.embedding(inputs) # [i/p:(1,batch)  o/p:(1,batch,embedding_size)]
        last_hidden = hidden[0] if self.decoder_type=='lstm' else hidden
        last_hidden = last_hidden.view(self.n_layers,last_hidden.size(1),last_hidden.size(2))
        last_hidden = last_hidden[-1]
        appearance_feats, appearance_weights = self.attention(last_hidden,appearance_feats) #(100,1536) #(100,28,1)
        motion_feats, motion_weights = self.attention(last_hidden,motion_feats) #(100,1536) #(100,28,1)
        object_feats,object_weights = self.attention(last_hidden,object_feats)
        context_vector = torch.cat((appearance_feats,motion_feats,object_feats),dim=1).unsqueeze(0) #(1,B,512*3)
        #print('context vector size :',context_vector.size())
        #print('embedded size :',embedded.size())
        if self.opt_EncoderCG:
            CGA,CGM,CGO = self.EncoderCG(vis_mem,lang_mem,context_vector)
            appearance_feats = appearance_feats * CGA
            motion_feats = motion_feats * CGM
            object_feats = object_feats *CGO
            context_vector = torch.cat((appearance_feats,motion_feats,object_feats),dim=1).unsqueeze(0)
        if self.opt_DecoderCG:
            SG,TG = self.DecoderCG(vis_mem,lang_mem,context_vector,embedded)
            context_vector = context_vector * SG
            embedded = embedded *TG
        
        input_combined = torch.cat((embedded,context_vector),dim=2)
        output, hidden = self.rnn(input_combined, hidden) # (1,100,512)
        output = output.squeeze(0) # (100,512)
        output = self.out(output) # (100,num_words)
        output = F.softmax(output, dim = 1) #(100,num_words)
        return output, hidden, appearance_weights,context_vector.squeeze(0),embedded .squeeze(0) #change
    
     
class THVC(nn.Module):
    
    def __init__(self,voc,cfg,path):
        super(THVC,self).__init__()

        self.voc = voc
        self.path = path
        self.cfg = cfg
        
        
        self.encoder = Encoder(cfg).to(cfg.device)
        self.enc_optimizer = optim.Adam(self.encoder.parameters(),lr=cfg.encoder_lr)

        self.decoder = RecurrentDecoder(cfg,voc).to(cfg.device)
        self.dec_optimizer = optim.Adam(self.decoder.parameters(),lr=cfg.decoder_lr,amsgrad=True)
        
        if cfg.opt_auxiliary_heads:
            self.aux_heads = Auxiliary_heads(cfg).to(cfg.device)
            self.auxhead_loss_fn = nn.BCEWithLogitsLoss()
            self.auxhead_optimizer = optim.Adam(self.aux_heads.parameters(),lr=cfg.aux_lr)
            
        self.teacher_forcing_ratio = cfg.teacher_forcing_ratio
        self.print_every = cfg.print_every
        self.clip = cfg.clip
        self.device = cfg.device
        if cfg.opt_param_init:
            self.init_params()
        self.epoch = 0
        
        self.visual_memory = torch.zeros(cfg.batch_size,cfg.projected_feature_size).to(self.device)
        self.target_memory = torch.zeros(cfg.batch_size,cfg.embedding_size).to(self.device)
        
        
    def init_params(self):
        for name, param in self.decoder.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
                #nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')

        
        
    def update_hyperparameters(self,cfg):
        
        
        self.enc_optimizer = optim.Adam(self.encoder.parameters(),lr=cfg.encoder_lr)
        
        self.dec_optimizer = optim.Adam(self.decoder.parameters(),lr=cfg.decoder_lr,amsgrad=True)
        self.teacher_forcing_ratio = cfg.teacher_forcing_ratio
        if cfg.opt_auxiliary_heads:
            self.auxhead_optimizer = optim.Adam(self.aux_heads.parameters(),lr=cfg.aux_lr)
        
        
    def load(self,encoder_path = 'Save/Meanpool_10.pt',decoder_path='Saved/SALSTM_10.pt'):
        if os.path.exists(encoder_path) and os.path.exists(decoder_path):
            self.encoder.load_state_dict(torch.load(encoder_path))
            self.decoder.load_state_dict(torch.load(decoder_path))
        else:
            print('File not found Error..')

    def save(self,encoder_path,decoder_path):
        if os.path.exists(encoder_path) and os.path.exists(decoder_path):
            
            torch.save(model.encoder.state_dict(),encoder_path)
            torch.save(model.decoder.state_dict(),decoder_path)
        else:
            print('Invalid path address given.')
            
    def train_epoch(self,dataloader,utils):
        '''
        Function to train the model for a single epoch.
        Args:
         Input:
            dataloader : the dataloader object.basically train dataloader object.
         Return:
             epoch_loss : Average single time step loss for an epoch
        '''
        total_loss = 0
        total_ac_loss = 0
        start_iteration = 1
        print_loss = 0
        print_ac_loss = 0
        iteration = 1
        self.encoder.train()
        self.decoder.train()
        for data in dataloader:
            appearance_features, targets, mask, max_length, _,motion_features,object_features,aux_obj,aux_act = data #change
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
            loss,ac_loss = self.train_iter(utils,appearance_features,motion_features,object_features,
                                   targets,mask,max_length,use_teacher_forcing,aux_obj,aux_act)
            print_loss += loss
            print_ac_loss += ac_loss
            total_loss += loss
            total_ac_loss += ac_loss
            
        # Print progress
            if iteration % self.print_every == 0:
                print_loss_avg = print_loss / self.print_every
                print_ac_loss_avg = print_ac_loss/self.print_every
                print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f};AC loss: {:.4f}".
                format(iteration, iteration / len(dataloader) * 100, print_loss_avg,print_ac_loss_avg))
                print_loss = 0
                print_ac_loss = 0
             
            
            iteration += 1 
        return total_loss/len(dataloader),total_ac_loss/len(dataloader)
        
        
    def train_iter(self,utils,input_variable,motion_variable,object_variable,target_variable,
                   mask,max_target_len,use_teacher_forcing,aux_obj,aux_act):
        '''
        Forward propagate input signal and update model for a single iteration. 
        
        Args:
        Inputs:
            input_variable : video mini-batch tensor; size = (B,T,F)
            motion_variable : video motion tensor; size = (B,T,F)
            target_variable : Ground Truth Captions;  size = (T,B); T will be different for different mini-batches
            mask : Masked tensor for Ground Truth;    size = (T,C)
            max_target_len : maximum lengh of the mini-batch; size = T
            use_teacher_forcing : binary variable. If True training uses teacher forcing else sampling.
            clip : clip the gradients to counter exploding gradient problem.
        Returns:
            iteration_loss : average loss per time step.
        '''
        
        self.enc_optimizer.zero_grad()
        self.dec_optimizer.zero_grad()
        if self.cfg.opt_auxiliary_heads:
            self.auxhead_optimizer.zero_grad()
        
        loss = 0
        ac_loss = 0
        total_loss = 0
        print_losses = []
        n_totals = 0
        
        input_variable = input_variable.to(self.device)
        motion_variable = motion_variable.to(self.device)
        object_variable = object_variable.to(self.device)
        
        input_variable,motion_variable,object_variable = self.encoder(input_variable,motion_variable,object_variable)  
        #print('input variable shape :',input_variable.size())
        #print('motion variable shape :',motion_variable.size())
        if self.cfg.opt_auxiliary_heads:
            aux_obj = aux_obj.to(self.device)
            aux_act = aux_act.to(self.device)
            object_auxout, action_auxout = self.aux_heads(motion_variable,object_variable)
            
        target_variable = target_variable.to(self.device)
        mask = mask.byte().to(self.device)
        
        # Forward pass through encoder
        decoder_input = torch.LongTensor([[self.cfg.SOS_token for _ in range(self.cfg.batch_size)]])
        decoder_input = decoder_input.to(self.device)
        decoder_hidden = torch.zeros(self.cfg.n_layers, self.cfg.batch_size,
                                      self.cfg.decoder_hidden_size).to(self.device)
        if self.cfg.decoder_type == 'lstm':
            decoder_hidden = (decoder_hidden,decoder_hidden)
            
        self.visual_memory.fill_(0)
        self.target_memory.fill_(0)
        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                hidden_last = decoder_hidden
                decoder_output, decoder_hidden,attn_weight,ct,et = self.decoder(decoder_input, decoder_hidden,
                                                input_variable.float(),motion_variable.float(),object_variable.float(),
                                                                               self.visual_memory.float(),
                                                                                self.target_memory.float())
                # Teacher forcing: next input comes from ground truth(data distribution)
                self.visual_memory = self.visual_memory + 1/(t+1)*ct.detach()
                self.target_memory = self.target_memory + 1/(t+1)*et.detach()
                decoder_input = target_variable[t].view(1, -1)            
                # decoder_output : (100,voc.num_words); target_variable[t] : (100); mask[t] : (100)
                mask_loss, nTotal = utils.maskNLLLoss(decoder_output.unsqueeze(0), target_variable[t], mask[t],self.device)
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden,attn_weight,ct,et = self.decoder(decoder_input,
                                            decoder_hidden,input_variable.float(),motion_variable.float(),
                                                                               object_variable.float(),
                                                                               self.visual_memory.float(),
                                                                               self.target_memory.float())
                # No teacher forcing: next input is decoder's own current output(model distribution)
                _, topi = decoder_output.squeeze(0).topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.cfg.batch_size)]])
                decoder_input = decoder_input.to(self.device)
                self.visual_memory = self.visual_memory + 1/(t+1)*ct.detach()
                self.target_memory = self.target_memory + 1/(t+1)*et.detach()
                
                # Calculate and accumulate loss
                mask_loss, nTotal = utils.maskNLLLoss(decoder_output, target_variable[t], mask[t],self.device)
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal

        ac_loss = self._calculate_AC_loss(attn_weight)
        fc_loss = self._calculate_Contrastive_Loss(input_variable)
        mc_loss = self._calculate_Contrastive_Loss(motion_variable)
        oc_loss = self._calculate_Contrastive_Loss(object_variable)
        
        #Closs = self.cfg.acl_weight*ac_loss + self.cfg.fcl_weight *fc_loss + self.cfg.ocl_weight *oc_loss
        
        
        Closs = self.cfg.acl_weight*ac_loss + self.cfg.fcl_weight *fc_loss + self.cfg.ocl_weight *oc_loss+self.cfg.mcl_weight*mc_loss
        
        if self.cfg.opt_auxiliary_heads:
            #calculate loss
            category_head_loss = 0 # No category information for MSVD
            object_head_loss = self.auxhead_loss_fn(object_auxout,aux_obj) 
            action_head_loss = self.auxhead_loss_fn(action_auxout,aux_act)
            Aloss = self.cfg.ohw*object_head_loss + self.cfg.ahw*action_head_loss
            total_loss = Closs + Aloss + loss
        else:
            total_loss = Closs + loss
            
        
        #total_loss = Closs + Aloss + loss
        # Perform backpropatation
        total_loss.backward()

        
        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)
            
        self.enc_optimizer.step()
        self.dec_optimizer.step()
        if self.cfg.opt_auxiliary_heads:
            self.auxhead_optimizer.step()
        
        
        return sum(print_losses) / n_totals, Closs.item()
    
    def _calculate_AC_loss(self,alphas):
        '''
        Calculate Attention-Coherent Loss.
        '''
        alphas = alphas.squeeze(2)
        alpha_next = alphas[:,1:]
        alpha_previous = alphas[:,:-1]
        ac_loss = torch.abs(alpha_next - alpha_previous).sum()
        
        return ac_loss 
    
    def _calculate_Contrastive_Loss(self,sequences):
        '''
        Calculate Frame Coherent Loss
        sequences: (B,28,1024)
        
        '''
        sequences_next = sequences[:,1:,:]
        sequences_previous = sequences[:,:-1,:]
        c_loss = torch.abs(sequences_next- sequences_previous).mean()
        return c_loss
        
    @torch.no_grad()
    def GreedyDecoding(self,features,motion_features,object_features,max_length=15):
        batch_size = features.size()[0]
        
        visual_memory = torch.zeros(batch_size,self.cfg.projected_feature_size).to(self.device)
        target_memory = torch.zeros(batch_size,self.cfg.embedding_size).to(self.device)
        
        features = features.to(self.device)
        motion_features = motion_features.to(self.device)
        object_features = object_features.to(self.device)
        
        
        features,motion_features,object_features = self.encoder(features,motion_features,object_features)  
        
        decoder_input = torch.LongTensor([[self.cfg.SOS_token for _ in range(batch_size)]]).to(self.device)
        decoder_hidden = torch.zeros(self.cfg.n_layers, batch_size,
                                      self.cfg.decoder_hidden_size).to(self.device)
        if self.cfg.decoder_type == 'lstm':
            decoder_hidden = (decoder_hidden,decoder_hidden)
        caption = []
        caption_prob = []
        attention_values = []
        for t in range(max_length):
            hidden_last = decoder_hidden
            #change
            decoder_output, decoder_hidden,attn_values,ct,et = self.decoder(decoder_input, 
                                                    decoder_hidden,features.float(),motion_features.float(),
                                                                           object_features.float(),
                                                                           visual_memory.float(),
                                                                           target_memory.float())
            
            visual_memory = visual_memory + 1/(t+1)*ct.detach()
            target_memory = target_memory + 1/(t+1)*et.detach()

            top_prob, topi = decoder_output.squeeze(0).topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]]).to(self.device)
            caption.append(topi.squeeze(1).cpu())
            caption_prob.append(top_prob.squeeze(1).cpu())
            attention_values.append(attn_values.squeeze(2))
        caption = torch.stack(caption,0).permute(1,0)
        caption_prob = torch.stack(caption_prob,0).permute(1,0)
        caps_text = []
        for dta in caption:
            tmp = []
            for token in dta:
                if token.item() not in self.voc.index2word.keys() or token.item()==2: # Remove EOS and bypass OOV
                    pass
                else:
                    tmp.append(self.voc.index2word[token.item()])
            tmp = ' '.join(x for x in tmp)
            caps_text.append(tmp)
        return caption,caps_text, torch.stack(attention_values,0).cpu().numpy(),caption_prob #change
    
    
    @torch.no_grad()
    def BeamDecoding(self,feats, motion_feats,width, alpha=0.,max_caption_len = 15):
        batch_size = feats.size(0)
        vocab_size = self.voc.num_words
        
        feats,motion_feats = self.encoder(feats,motion_feats)  
        

        vfunc = np.vectorize(lambda t: self.voc.index2word[t]) # to transform tensors to words
        rfunc = np.vectorize(lambda t: '' if t == 'EOS' else t) # to transform EOS to null string
        lfunc = np.vectorize(lambda t: '' if t == 'SOS' else t) # to transform SOS to null string
        pfunc = np.vectorize(lambda t: '' if t == 'PAD' else t) # to transform PAD to null string

        hidden = torch.zeros(self.cfg.n_layers, batch_size, self.cfg.decoder_hidden_size).to(self.device)
        if self.cfg.decoder_type == 'lstm':
            hidden = (hidden,hidden)
        
        input_list = [ torch.cuda.LongTensor(1, batch_size).fill_(self.cfg.SOS_token) ]
        hidden_list = [ hidden ]
        cum_prob_list = [ torch.ones(batch_size).cuda() ]
        cum_prob_list = [ torch.log(cum_prob) for cum_prob in cum_prob_list ]
        EOS_idx = self.cfg.EOS_token

        output_list = [ [[]] for _ in range(batch_size) ]
        for t in range(max_caption_len + 1):
            beam_output_list = [] # width x ( 1, 100 )
            normalized_beam_output_list = [] # width x ( 1, 100 )
            if self.cfg.decoder_type == "lstm":
                beam_hidden_list = ( [], [] ) # 2 * width x ( 1, 100, 512 )
            else:
                beam_hidden_list = [] # width x ( 1, 100, 512 )
            next_output_list = [ [] for _ in range(batch_size) ]
            assert len(input_list) == len(hidden_list) == len(cum_prob_list)
            for i, (input, hidden, cum_prob) in enumerate(zip(input_list, hidden_list, cum_prob_list)):
                output, next_hidden, _,_,_ = self.decoder(input, hidden, feats,motion_feats) # need to check

                caption_list = [ output_list[b][i] for b in range(batch_size)]
                EOS_mask = [ 0. if EOS_idx in [ idx.item() for idx in caption ] else 1. for caption in caption_list ]
                EOS_mask = torch.cuda.FloatTensor(EOS_mask)
                EOS_mask = EOS_mask.unsqueeze(1).expand_as(output)
                output = EOS_mask * output

                output += cum_prob.unsqueeze(1)
                beam_output_list.append(output)

                caption_lens = [ [ idx.item() for idx in caption ].index(EOS_idx) + 1 if EOS_idx in [ idx.item() for idx in caption ] else t + 1 for caption in caption_list ]
                caption_lens = torch.cuda.FloatTensor(caption_lens)
                normalizing_factor = ((5 + caption_lens) ** alpha) / (6 ** alpha)
                normalizing_factor = normalizing_factor.unsqueeze(1).expand_as(output)
                normalized_output = output / normalizing_factor
                normalized_beam_output_list.append(normalized_output)
                if self.cfg.decoder_type == "lstm":
                    beam_hidden_list[0].append(next_hidden[0])
                    beam_hidden_list[1].append(next_hidden[1])
                else:
                    beam_hidden_list.append(next_hidden)
            beam_output_list = torch.cat(beam_output_list, dim=1) # ( 100, n_vocabs * width )
            normalized_beam_output_list = torch.cat(normalized_beam_output_list, dim=1)
            beam_topk_output_index_list = normalized_beam_output_list.argsort(dim=1, descending=True)[:, :width] # ( 100, width )
            topk_beam_index = beam_topk_output_index_list // vocab_size # ( 100, width )
            topk_output_index = beam_topk_output_index_list % vocab_size # ( 100, width )

            topk_output_list = [ topk_output_index[:, i] for i in range(width) ] # width * ( 100, )
            if self.cfg.decoder_type == "lstm":
                topk_hidden_list = (
                    [ [] for _ in range(width) ],
                    [ [] for _ in range(width) ]) # 2 * width * (1, 100, 512)
            else:
                topk_hidden_list = [ [] for _ in range(width) ] # width * ( 1, 100, 512 )
            topk_cum_prob_list = [ [] for _ in range(width) ] # width * ( 100, )
            for i, (beam_index, output_index) in enumerate(zip(topk_beam_index, topk_output_index)):
                for k, (bi, oi) in enumerate(zip(beam_index, output_index)):
                    if self.cfg.decoder_type == "lstm":
                        topk_hidden_list[0][k].append(beam_hidden_list[0][bi][:, i, :])
                        topk_hidden_list[1][k].append(beam_hidden_list[1][bi][:, i, :])
                    else:
                        topk_hidden_list[k].append(beam_hidden_list[bi][:, i, :])
                    topk_cum_prob_list[k].append(beam_output_list[i][vocab_size * bi + oi])
                    next_output_list[i].append(output_list[i][bi] + [ oi ])
            output_list = next_output_list

            input_list = [ topk_output.unsqueeze(0) for topk_output in topk_output_list ] # width * ( 1, 100 )
            if self.cfg.decoder_type == "lstm":
                hidden_list = (
                    [ torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list[0] ],
                    [ torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list[1] ]) # 2 * width * ( 1, 100, 512 )
                hidden_list = [ ( hidden, context ) for hidden, context in zip(*hidden_list) ]
            else:
                hidden_list = [ torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list ] # width * ( 1, 100, 512 )
            cum_prob_list = [ torch.cuda.FloatTensor(topk_cum_prob) for topk_cum_prob in topk_cum_prob_list ] # width * ( 100, )

        SOS_idx = self.cfg.SOS_token
        outputs = [ [ SOS_idx ] + o[0] for o in output_list ]
        
        outputs = [[torch.tensor(y) for y in x] for x in outputs]
        outputs = [[y.item() for y in x] for x in outputs]
        
        captions = vfunc(outputs)
        captions = rfunc(captions)
        captions = lfunc(captions)
        captions = pfunc(captions)
        caps_text = []

        for eee in captions:
            caps_text.append(' '.join(x for x in eee).strip())
        
        
        return caps_text

