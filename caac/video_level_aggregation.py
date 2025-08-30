# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 19:03:25 2025

@author: pothugun
"""

import os
import torch
import torch.nn.functional as F
import argparse

# path1 = 'C:/Users/pothugun/Downloads/JOC/Video_Files/Pickles'
# out_path = 'C:/Users/pothugun/Downloads/JOC/Video_Files/Co_Attn_Final'

def weighted_sharing(embedding):
    stacked = torch.cat(embedding, dim=0)  
    scores = stacked.mean(dim=1)  
    weights = F.softmax(scores, dim=0) 
    weighted = stacked * weights.unsqueeze(1) 
    final_output = weighted.sum(dim=0, keepdim=True)
    return final_output

def video_embed(path1, out_path):
    for file in os.listdir(path1):
        tensor_list = []
        new_path = os.path.join(path1, file)
        print(new_path)
        try:
          for file1 in os.listdir(new_path):
              tensor_list.append(torch.load(f'{new_path}/{file1}'))
              print('one')
          co_attn_video = weighted_sharing(tensor_list)
          print('two)
          if not os.path.isdir(out_path):
                os.makedirs(out_path)
          torch.save(co_attn_video, f'{out_path}/{file}.pt')
        except:
          continue
        
def main():
    parser = argparse.ArgumentParser(description = 'JOC Encoded Tensors')
        
    parser.add_argument('--input', '-i', required = True, help = 'Input CLS Tokens')
    parser.add_argument('--output', '-o', required = True, help = 'Output Video CLS')
    
    args = parser.parse_args()
    
    try:
        video_embed(args.input, args.output)
        print('success')
    except Exception as e:
        print(f'Error occured: {e}')
        
if __name__ == "__main__":
    main()