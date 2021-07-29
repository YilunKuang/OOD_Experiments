from transformers import AutoTokenizer, AutoModel, BartForConditionalGeneration, BartTokenizerFast
from datasets import load_dataset
import torch as torch
from tqdm import tqdm
from datasets import load_metric
import numpy as np
import argparse

# Example command line input
# python PPL_bart_xsum.py --dataset_name xsum --model_name_or_path /scratch/yk2516/OOD_Text_Generation/BART-Wikihow/checkpoint-final

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", help="choose a dataset from wikihow, gigaword, big_patent, xsum, cnn_dailymail, billsum",
                    type=str)
parser.add_argument("--model_name_or_path", help="choose a model_checkpoint from either a1noack/bart-large-gigaword, \
                                                                                        facebook/bart-large, \
                                                                                        or /scratch/yk2516/OOD_Text_Generation/BART-Wikihow/checkpoint-final",
                    type=str)
parser.add_argument('--test_case', help="if called, then we only iterate over the first 20 examples without \
                                         going through the whole test set. This is for debugging only.", action='store_true')

args = parser.parse_args()
dataset_lst = ['wikihow', 'gigaword', 'big_patent', 'xsum', 'cnn_dailymail', 'billsum']
model_checkpoint_lst = ['a1noack/bart-large-gigaword', 'facebook/bart-large', '/scratch/yk2516/OOD_Text_Generation/BART-Wikihow/checkpoint-final']

def main():
    if args.model_name_or_path not in model_checkpoint_lst:
        raise ValueError('Please enter a valid model_name_or_path')
    # model_checkpoint = 'a1noack/bart-large-gigaword' #'facebook/bart-large' #'a1noack/bart-large-gigaword' #'facebook/bart-base'  #'facebook/bart-large-cnn'
    model_checkpoint = args.model_name_or_path
    tokenizer = BartTokenizerFast.from_pretrained(model_checkpoint)
    model = BartForConditionalGeneration.from_pretrained(model_checkpoint, return_dict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    if args.dataset_name == 'wikihow':
        test = load_dataset("wikihow", "all", data_dir="/scratch/yk2516/OOD_Text_Generation/wikihow_manual", split='test', cache_dir='/scratch/yk2516/cache/')
        input_column = 'text'
    if args.dataset_name == 'gigaword':
        test = load_dataset('gigaword', split='test',cache_dir='/scratch/yk2516/cache/')
        input_column = 'document'
    if args.dataset_name == 'big_patent':
        test = load_dataset('big_patent', "g", split='test',cache_dir='/scratch/yk2516/cache/')
        input_column = 'description'
    if args.dataset_name == 'xsum':
        test = load_dataset("xsum",cache_dir='/scratch/yk2516/cache/', split='test')
        input_column = 'document'
    if args.dataset_name == 'cnn_dailymail':
        test = load_dataset(path="cnn_dailymail",name='3.0.0',split='test',cache_dir='/scratch/yk2516/cache/')
        input_column = 'article'
    if args.dataset_name == 'billsum':
        test = load_dataset("billsum", split='test',cache_dir='/scratch/yk2516/cache/')
        input_column = 'text'
    if args.dataset_name not in dataset_lst:
        raise ValueError('Please enter a valid dataset name')
    
    print(model_checkpoint + ' on ' + args.dataset_name)
    print(test)
    print(model.config.max_length)
    encodings =  tokenizer(test[input_column], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device)

    model = model.to(device)
    model.eval()
    number_beams = 8
    print("Input ids size", encodings['input_ids'].shape)
    ids = encodings['input_ids'].cpu().detach().numpy()
    attention_ids = encodings['attention_mask'].cpu().detach().numpy()
    log_sent = []
    num_words = []
    print("number of samples:", ids.shape[0])
    
    if args.test_case:
        n_sample = 20
    else:
        n_sample = ids.shape[0]
        
    for i in range(n_sample):
        input_id = np.array([ids[i]])
        input_id = torch.from_numpy(input_id).to(device)
        attention_id = np.array([attention_ids[i]])
        attention_id = torch.from_numpy(attention_id).to(device)
        with torch.no_grad():
            result = model.generate(input_ids=input_id, attention_mask=attention_id, num_beams=number_beams, return_dict_in_generate=True, max_length=model.config.max_length, output_scores=True, output_attentions=True)
        
        all = []
        
        for batch_num in range(0, result.scores[0].shape[0], number_beams):
            max_score = torch.tensor(-1*1e6, dtype=torch.float).to(device)
            for beam_num in range(number_beams):
                # print([torch.max(result.scores[-1][batch_num+beam_num]), max_score])
                max_score = torch.max(torch.stack([torch.max(result.scores[-1][batch_num+beam_num]), max_score]))
            log_sent.append(max_score)
            num_words.append(result.sequences.shape[1])
    
    print(log_sent)
    print(torch.stack(log_sent).sum())
    total_words = sum(num_words)
    print(total_words)
    print("Perplexity: ", torch.exp((-1*(torch.stack(log_sent).sum()))/total_words))

if __name__ == "__main__":
    main()
