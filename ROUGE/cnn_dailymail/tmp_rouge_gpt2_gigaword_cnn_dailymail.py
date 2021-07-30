# -*- coding: utf-8 -*-
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset,load_metric
import torch as torch
from tqdm import tqdm
import numpy as np
import pickle

# changes:
# 1. model_checkpoint
# 2. load_dataset
# 3. encodings, rouge.add

model_checkpoint = '/scratch/yk2516/OOD_Text_Generation/checkpoint-36000'
tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
model = GPT2LMHeadModel.from_pretrained(model_checkpoint, pad_token_id=tokenizer.eos_token_id, return_dict=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
rouge = load_metric('rouge', cache_dir='/scratch/yk2516/OOD_Text_Generation/cache_rouge')
test = load_dataset("big_patent", "g", split='test',cache_dir='/scratch/yk2516/cache/')

print(device)
print(test)
print(model.config.max_length)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
ignore_idx = tokenizer.pad_token_id
encodings = tokenizer(test['description'], return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

model.resize_token_embeddings(len(tokenizer))
model = model.to(device)
model.eval()

print("Input ids size", encodings['input_ids'].shape)
ids = encodings['input_ids'].cpu().detach().numpy()
attention_ids = encodings['attention_mask'].cpu().detach().numpy()
print("number of samples:", ids.shape[0])

lst_gen_sum = []
for i in range(2000):
    print(i)
    input_id = np.array([ids[i]])
    input_id = torch.from_numpy(input_id).to(device)
    attention_id = np.array([attention_ids[i]])
    attention_id = torch.from_numpy(attention_id).to(device)

    with torch.no_grad():
        output = model.generate(input_ids=input_id, 
                                attention_mask=attention_id, 
                                #num_beams=number_beams, 
                                return_dict_in_generate=True, 
                                max_length= 1024, 
                                output_scores=True, 
                                output_attentions=True)

    generated_summary = tokenizer.batch_decode(output[0], skip_special_tokens=True) #, clean_up_tokenization_spaces=False)
    lst_gen_sum.append(generated_summary)
    rouge.add(prediction=generated_summary[0], reference=test['description'][i])
    
with open('/scratch/yk2516/OOD_Text_Generation/GPT2-CNN/gen_sum_rouge_gpt2_cnn_dailymail.pkl', 'wb') as f:
    pickle.dump(lst_gen_sum, f)
    
score = rouge.compute()
print('Rouge: ')
print(score)



