# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 02:56:27 2025

@author: kpoth
"""

import numpy as np
import pandas as pd
#from scipy.special import softmax
from scipy.stats import entropy
import os
import argparse

def compute_mean_entropy(matrix, eps=1e-12):
    matrix = np.clip(matrix, eps, 1)
    row_sums = matrix.sum(axis=1, keepdims=True)
    probs = matrix / row_sums
    entropies = entropy(probs.T)  # scipy computes along columns, so transpose
    return np.mean(entropies)

def row_normalize(x):
    x = np.asarray(x)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def compute_attention_entropies(pathIn, pathOut):
    videoids_scores = []

    for folder in os.listdir(pathIn):
        
        text_to_frame_path = os.path.join(pathIn, folder,'T2F')
        frame_to_text_path = os.path.join(pathIn, folder,'F2T')
        
        entropy_t2f_list = []
        entropy_f2t_list = []
        
        for i in range(len(os.listdir(text_to_frame_path))):
            try:
                text_to_frame = pd.read_csv(f'{text_to_frame_path}/text_to_frame{i}.csv')
                frame_to_text = pd.read_csv(f'{frame_to_text_path}/frame_to_text{i}.csv')
                text_to_frame = text_to_frame.drop(columns = ['Unnamed: 0'])
                frame_to_text = frame_to_text.drop(columns = ['Unnamed: 0'])
                
                t2f_probs = row_normalize(text_to_frame.values)
                f2t_probs = row_normalize(frame_to_text.values)
                
                entropy_t2f = compute_mean_entropy(t2f_probs)
                entropy_f2t = compute_mean_entropy(f2t_probs)

            except Exception as e:
                # If error, set to max entropy for uniform distribution
                entropy_t2f = np.log(text_to_frame.shape[1]) if text_to_frame.shape[1] > 0 else 0
                entropy_f2t = np.log(frame_to_text.shape[1]) if frame_to_text.shape[1] > 0 else 0

            entropy_t2f_list.append(entropy_t2f)
            entropy_f2t_list.append(entropy_f2t)

        # Aggregate entropy across time
        entropy_t2f_avg = np.mean(entropy_t2f_list)
        entropy_f2t_avg = np.mean(entropy_f2t_list)

        videoids_scores.append([folder, entropy_t2f_avg, entropy_f2t_avg])

    df = pd.DataFrame(videoids_scores, columns=['video_id', 'entropy_t2f', 'entropy_f2t'])
    df.to_csv(f'{pathOut}/attention_entropies_new.csv', index=False)
    
# pathIn = 'C:/Users/kpoth/Downloads/JOC/Testing/Attention_Files'
# pathOut = 'C:/Users/kpoth/Downloads/JOC/Testing'
# compute_attention_entropies(pathIn, pathOut)

def main():
    parser = argparse.ArgumentParser(description = 'JOC Encoded Tensors')
        
    parser.add_argument('--input1', '-i', required = True, help = 'Input Attn Files')
    parser.add_argument('--input2', '-o', required = True, help = 'Output DF path')
    
    args = parser.parse_args()
    
    try:
        compute_attention_entropies(args.input1, args.input2)
        print('success')
    except Exception as e:
        print(f'Error occured: {e}')

if __name__ == "__main__":
    main()