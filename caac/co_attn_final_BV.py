# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 10:18:17 2025

@author: kpoth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, ViTModel, ViTImageProcessor
from PIL import Image
import numpy as np
import pandas as pd
import os
import argparse

hidden_dim = 768
num_heads = 8

vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', output_attentions=True)
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
out_path = '/mnt/gs21/scratch/pothugun/Nov25_28_Output/'

def text_output(text, pick_path, attn_path, cnt):
    model_name = f"{out_path}/bert-1024-extended"
    tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
    model = BertModel.from_pretrained(model_name, output_attentions=True, local_files_only=True)
    special_characters = ",!?...-_*@#_/\\\"'():;~|^=[]{}<>+=`"
    cleaned_sentence = text.translate(str.maketrans("", "", special_characters))
    inputs = tokenizer(cleaned_sentence, max_length = 1024, truncation = True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    hidden_states = outputs.last_hidden_state  
    attention_matrices = outputs.attentions
    tokenlabels = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist())
    
    all_layers_attention = [layer[0].detach().numpy() for layer in attention_matrices]  
    all_layers_avg = sum(layer.mean(axis=0) for layer in all_layers_attention) / len(all_layers_attention)
    df_all_layers = pd.DataFrame(all_layers_avg, index=tokenlabels, columns=tokenlabels)
    new_attn_path = os.path.join(attn_path, 'text_SA')
    if not os.path.isdir(new_attn_path):
          os.makedirs(new_attn_path)
    df_all_layers.to_csv(f'{new_attn_path}/text{cnt}_SA.csv')
    clstext = hidden_states[:,0,:]
    new_pick_path = os.path.join(pick_path, 'Text')
    if not os.path.isdir(new_pick_path):
          os.makedirs(new_pick_path)
    torch.save(clstext, f'{new_pick_path}/CLS{cnt}.pt')
    return outputs, tokenlabels


def image_embeddings(image_paths):
    images = Image.open(image_paths).convert('RGB')
    image_inputs = feature_extractor(images=images, return_tensors='pt')
    with torch.no_grad():
        image_outputs = vit_model(**image_inputs)
    return image_outputs

def frame_output(image_paths, pick_path, attn_path, cnt):
    tensor_list = []
    attnmap_list = []
    for a in image_paths:
        try:
            outputs = image_embeddings(a)
            hidden_states = outputs.last_hidden_state
            
            hidden_dim = hidden_states.shape[-1]  
            query_projection = torch.nn.Linear(hidden_dim, hidden_dim)
            key_projection = torch.nn.Linear(hidden_dim, hidden_dim)
            value_projection = nn.Linear(hidden_dim, hidden_dim)
            
            Q = query_projection(hidden_states)  
            K = key_projection(hidden_states)
            V = value_projection(hidden_states)
            attention_scores = torch.matmul(Q, K.transpose(-1, -2))
            d_k = hidden_dim ** 0.5
            attention_scores = attention_scores / d_k
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_map = attention_weights[0].detach().numpy()
            self_attended_tensor = torch.matmul(attention_weights, V)
            tensor_list.append(self_attended_tensor)
            attnmap_list.append(attention_map)
        except:
            continue
    array_stack = np.stack(attnmap_list, axis=0)
    mean_array = np.mean(array_stack, axis=0)
    df = pd.DataFrame(mean_array)
    new_path = os.path.join(attn_path, 'Frame_SA')
    if not os.path.isdir(new_path):
          os.makedirs(new_path)
    df.to_csv(f'{new_path}/frame{cnt}_SA.csv')
    tensor_stack = torch.stack(tensor_list, dim=0).squeeze(1)
    clsframe = tensor_stack[:,0,:]
    meanclsframe = torch.mean(clsframe, dim = 0).unsqueeze(dim = 0)
    new_pick_path = os.path.join(pick_path, 'Frame')
    if not os.path.isdir(new_pick_path):
          os.makedirs(new_pick_path)
    torch.save(meanclsframe, f'{new_pick_path}/CLS{cnt}.pt')
    #return tensor_stack

def single_sentence_image(text_features, image_features):
    text_to_image_attn_raw = torch.matmul(text_features.squeeze(0), image_features.squeeze(0).transpose(0, 1)) / (hidden_dim ** 0.5)
    image_to_text_attn_raw = torch.matmul(image_features.squeeze(0), text_features.squeeze(0).transpose(0, 1)) / (hidden_dim ** 0.5)
    
    #combined_raw = torch.stack([raw_sum1, raw_sum2], dim=1)
    #normalized = combined_raw / combined_raw.sum(dim=1, keepdim=True)
    
    text_to_image_attn = F.softmax(text_to_image_attn_raw, dim=1)  
    image_to_text_attn = F.softmax(image_to_text_attn_raw, dim=1)  
    
    attended_text = torch.matmul(image_to_text_attn, text_features.squeeze(0))
    attended_image = torch.matmul(text_to_image_attn, image_features.squeeze(0))
    return attended_text, attended_image, text_to_image_attn_raw, image_to_text_attn_raw

def combined_coattn(attended_text, attended_image):
    T, d = attended_text.shape
    P = attended_image.shape[0]
    
    proj_text = nn.Linear(d, 1, bias=False)
    
    attn_scores = proj_text(attended_text)  
    attn_weights = F.softmax(attn_scores, dim=0)  
    
    pooled_text = torch.sum(attn_weights * attended_text, dim=0, keepdim=True)  # (1, d)
    pooled_text = pooled_text.expand(P, d)
    gate = torch.sigmoid(pooled_text + attended_image)
    combined = gate * pooled_text + (1 - gate) * attended_image
    joint_embedding = combined.mean(dim=0)
    return joint_embedding

def sentence_frames(sentence, frame_list, pick_path, attn_path, count):
    t2f_embed = []
    f2t_embed = []
    co_attn_embed = []
    t2f_attn_list = []
    f2t_attn_list = []
    text_outputs, token_labels = text_output(sentence, pick_path, attn_path, count)
    text_features = text_outputs.last_hidden_state
    frame_output(frame_list, pick_path, attn_path, count)
    for a in frame_list:
        try:
            image_outputs = image_embeddings(a)
            image_features = image_outputs.last_hidden_state
            attended_text, attended_image, text_to_image_attn_raw, image_to_text_attn_raw = single_sentence_image(text_features, image_features)
            combined_sig = combined_coattn(attended_text, attended_image).unsqueeze(0)
            
            cls_t2f = attended_image[0].unsqueeze(0)
            cls_f2t = attended_text[0].unsqueeze(0)
            
            t2f_embed.append(cls_t2f)
            f2t_embed.append(cls_f2t)
            co_attn_embed.append(combined_sig)
            t2f_attn_list.append(text_to_image_attn_raw)
            f2t_attn_list.append(image_to_text_attn_raw)
        except:
            continue
    return t2f_embed, f2t_embed, co_attn_embed, t2f_attn_list, f2t_attn_list, token_labels
    
def norm_t2f(attn_list):
    text_frame_scores = [a.mean(dim=1, keepdim=True) for a in attn_list]
    text_frame_matrix = torch.cat(text_frame_scores, dim=1)
    row_sums = text_frame_matrix.sum(dim=1, keepdim=True)
    normalized_text_frame = text_frame_matrix / row_sums
    return normalized_text_frame

def norm_f2t(attn_list):
    frame_text_scores = [f.mean(dim=0, keepdim=True) for f in attn_list]
    frame_text_matrix = torch.cat(frame_text_scores, dim=0)
    row_sums = frame_text_matrix.sum(dim=1, keepdim=True)  
    normalized_frame_text = frame_text_matrix / row_sums
    return normalized_frame_text

def weighted_sharing(embedding):
    stacked = torch.cat(embedding, dim=0)  
    scores = stacked.mean(dim=1)  
    weights = F.softmax(scores, dim=0) 
    weighted = stacked * weights.unsqueeze(1) 
    final_output = weighted.sum(dim=0, keepdim=True)
    return final_output

def writing_single_sentence_frames(sentence, frame_list, pick_path, attn_path, count):
    t2f_embed, f2t_embed, co_attn_embed, t2f_attn_list, f2t_attn_list, token_labels = sentence_frames(sentence, frame_list, pick_path, attn_path, count)
    
    t2f_attn = norm_t2f(t2f_attn_list)
    t2f_attn_np = t2f_attn.detach().numpy()
    dft2f = pd.DataFrame(t2f_attn_np, index = token_labels) 
    
    f2t_attn = norm_f2t(f2t_attn_list)
    f2t_attn_np = f2t_attn.detach().numpy()
    dff2t = pd.DataFrame(f2t_attn_np, columns = token_labels) 
    
    text_path = os.path.join(attn_path, 'F2T')
    if not os.path.isdir(text_path):
        os.makedirs(text_path)
    dff2t.to_csv(f'{text_path}/frame_to_text{count}.csv')
    
    frame_path = os.path.join(attn_path, 'T2F')
    if not os.path.isdir(frame_path):
        os.makedirs(frame_path)
    dft2f.to_csv(f'{frame_path}/text_to_frame{count}.csv')
    
    t2f_final = weighted_sharing(t2f_embed)
    f2t_final = weighted_sharing(f2t_embed)
    coattn_final = weighted_sharing(co_attn_embed)
    
    coattn_t2f = os.path.join(pick_path, 'T2F')
    if not os.path.isdir(coattn_t2f):
          os.makedirs(coattn_t2f)
    torch.save(t2f_final, f'{coattn_t2f}/CLS{count}.pt')
    
    coattn_f2t = os.path.join(pick_path, 'F2T')
    if not os.path.isdir(coattn_f2t):
          os.makedirs(coattn_f2t)
    torch.save(f2t_final, f'{coattn_f2t}/CLS{count}.pt')
    
    coattn_path = os.path.join(pick_path, 'Co_Attn')
    if not os.path.isdir(coattn_path):
          os.makedirs(coattn_path)
    torch.save(coattn_final, f'{coattn_path}/CLS{count}.pt')

def single_video(txpath, framespath, pickpath, attentionpath):
    file_mod = os.path.basename(txpath).split('.')[0]
    df1 = pd.read_csv(txpath)
    count = 0
    df1['start_time'] = df1['start_time'].fillna(0)
    df1['end_time'] = df1['end_time'].fillna(0)
    
    attn_path = os.path.join(attentionpath, file_mod)
    
    for i in range(len(df1)):
        if len(df1.loc[i, 'sentences']) == 0:
            continue
        sentence1 = df1.loc[i, 'sentences']
        frame_path1 = os.path.join(framespath, file_mod)
        starttime = int(df1.loc[i, 'start_time'])
        endtime = int(df1.loc[i, 'end_time'])
        if endtime == starttime:
            endtime = starttime + 1
        
        endtime = endtime + 1
        
        frame_list = [f'{frame_path1}/frame{j}.jpg' for j in list(range(starttime, endtime))]
        new_pickpath = os.path.join(pickpath, file_mod)
        writing_single_sentence_frames(sentence1, frame_list, new_pickpath, attn_path, count)
        count = count + 1

# tx_path = 'C:/Users/kpoth/Downloads/JOC/Video_Files/shot_tx_sorted/KP/3jPGl_nADAA.csv'
# frames_path = 'C:/Users/kpoth/Downloads/JOC/Video_Files/HPC/Video_Frames/KP' 
# pick_path = 'C:/Users/kpoth/Downloads/JOC/Video_Files/Checking_Errors/Pickles'
# attention_path = 'C:/Users/kpoth/Downloads/JOC/Video_Files/Checking_Errors/Attention_Files'

# single_video(tx_path, frames_path, pick_path, attention_path)

def main():
    parser = argparse.ArgumentParser(description = 'JOC Encoded Tensors')
        
    parser.add_argument('--input1', '-i1', required = True, help = 'Input Text Path')
    parser.add_argument('--input2', '-i2', required = True, help = 'Input Frames Path')
    parser.add_argument('--output1', '-o1', required = True, help = 'Output tensor Path')
    parser.add_argument('--output2', '-o2', required = True, help = 'Output Attention Path')
    
    args = parser.parse_args()
    
    try:
        single_video(args.input1, args.input2, args.output1, args.output2)
        print('success')
    except Exception as e:
        print(f'Error occured: {e}')
        
if __name__ == "__main__":
    main()