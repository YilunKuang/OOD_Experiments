from transformers import AutoTokenizer, AutoModel, BartForConditionalGeneration, BartTokenizerFast
from datasets import load_dataset
import torch as torch
from tqdm import tqdm
from datasets import load_metric
import numpy as np


def main():
    model_checkpoint = 'a1noack/bart-large-gigaword' #'facebook/bart-large' #'a1noack/bart-large-gigaword' #'facebook/bart-base'  #'facebook/bart-large-cnn'
    tokenizer = BartTokenizerFast.from_pretrained(model_checkpoint)
    model = BartForConditionalGeneration.from_pretrained(model_checkpoint, return_dict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    # test = load_dataset("wikihow", "all", data_dir="/scratch/nm3571", split='test')
    # test = load_dataset("gigaword", split='test')
    # test = load_dataset("big_patent", "g", split='test')
    # test = load_dataset("xsum",cache_dir='/scratch/yk2516/cache/', split='test[:]')
    # test = load_dataset("cnn_daily",cache_dir='/scratch/yk2516/cache/', split='test[:]')
    # test = load_dataset("cnn_dailymail", name='3.0.0',download_mode="force_redownload",split='test', cache_dir='/scratch/yk2516/cache/')
    test = load_dataset(path="cnn_dailymail",name='3.0.0',split='test')
    
    print(test)
    print(model.config.max_length)
    encodings =  tokenizer(test['article'], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device)

    model = model.to(device)
    model.eval()
    number_beams = 8
    # print(encodings['input_ids'])
    # print(encodings['input_ids'].cpu().detach().numpy())
    # print(torch.from_numpy(encodings['input_ids'].cpu().detach().numpy()).to(device))
    # print(type(encodings['input_ids'][0].item()))
    # print("Wikihow vocab size: ", result.scores[0].shape[1])
    print("Input ids size", encodings['input_ids'].shape)
    ids = encodings['input_ids'].cpu().detach().numpy()
    attention_ids = encodings['attention_mask'].cpu().detach().numpy()
    log_sent = []
    num_words = []
    print("number of samples:", ids.shape[0])
    for i in range(ids.shape[0]):
        input_id = np.array([ids[i]])
        input_id = torch.from_numpy(input_id).to(device)
        attention_id = np.array([attention_ids[i]])
        attention_id = torch.from_numpy(attention_id).to(device)
        with torch.no_grad():
            result = model.generate(input_ids=input_id, attention_mask=attention_id, num_beams=number_beams, return_dict_in_generate=True, max_length=model.config.max_length, output_scores=True, output_attentions=True)
        
        all = []
        # print(result.sequences.shape)
        
        for batch_num in range(0, result.scores[0].shape[0], number_beams):
            # lls = torch.tensor(0, dtype=torch.float)
            # print(batch_num)
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
