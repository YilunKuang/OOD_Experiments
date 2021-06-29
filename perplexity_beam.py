from transformers import AutoTokenizer, AutoModel, BartForConditionalGeneration, BartTokenizerFast
from datasets import load_dataset
import torch as torch
from tqdm import tqdm
from datasets import load_metric
import numpy as np

# def main():
#     model_checkpoint = 'a1noack/bart-large-gigaword' # 'facebook/bart-large-cnn' #'facebook/bart-base'
#     tokenizer = BartTokenizerFast.from_pretrained(model_checkpoint)
#     model = BartForConditionalGeneration.from_pretrained(model_checkpoint, return_dict=True)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(device)
#     test = load_dataset("gigaword", split='test[:20]')
#     print(test)
#     print(model.config.max_length)
#     encodings =  tokenizer(test['document'], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device)

#     model = model.to(device)
#     model.eval()
#     number_beams = 8
#     with torch.no_grad():
#         result = model.generate(encodings['input_ids'],  num_beams=number_beams, return_dict_in_generate=True, max_length=model.config.max_length, output_scores=True, output_attentions=True)
    
#     all = []
#     log_sent = []
#     print(result.sequences.shape)
#     print("Gigaword vocab size: ", result.scores[0].shape[1])
#     print("Input ids size", encodings['input_ids'].shape)
#     for batch_num in range(0, result.scores[0].shape[0], number_beams):
#         # lls = torch.tensor(0, dtype=torch.float)
#         # print(batch_num)
#         max_score = torch.tensor(-1*1e6, dtype=torch.float).to(device)
#         for beam_num in range(number_beams):
#             print([torch.max(result.scores[-1][batch_num+beam_num]), max_score])
#             max_score = torch.max(torch.stack([torch.max(result.scores[-1][batch_num+beam_num]), max_score]))
#         log_sent.append(max_score)
        
#     print(log_sent)
#     print(torch.stack(log_sent).sum())
#     print(torch.exp((-1*(torch.stack(log_sent).sum()))/result.sequences.shape[1]))
#     # print(result.scores)
#     # print(result.scores[0].shape)
#     # print(max(result.scores[-1][-1]))
#     # print(max(result.scores[-2][-1]))
#     # print(max(result.scores[-3][-1]))
#     # print(max(result.scores[-4][-1]))
#     # print(result.sequences_scores[-1])
#     # print(result.sequences_scores[-1].sum())

#     # with torch.no_grad():
#     #     output = model(summary_ids, labels=summary_ids)

    

# if __name__ == "__main__":
#     main()


# def main():
#     model_checkpoint = 'a1noack/bart-large-gigaword' # 'facebook/bart-large-cnn' #'facebook/bart-base'
#     tokenizer = BartTokenizerFast.from_pretrained(model_checkpoint)
#     model = BartForConditionalGeneration.from_pretrained(model_checkpoint, return_dict=True)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(device)
#     test = load_dataset("wikihow", "all", data_dir="/scratch/nm3571", split='test[:20]')
#     print(test)
#     print(model.config.max_length)
#     encodings =  tokenizer(test['text'], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device)

#     model = model.to(device)
#     model.eval()
#     number_beams = 8
#     with torch.no_grad():
#         result = model.generate(encodings['input_ids'],  num_beams=number_beams, return_dict_in_generate=True, max_length=model.config.max_length, output_scores=True, output_attentions=True)
    
#     all = []
#     log_sent = []
#     print(result.sequences.shape)
#     print("Wikihow vocab size: ", result.scores[0].shape[1])
#     print("Input ids size", encodings['input_ids'].shape)
#     for batch_num in range(0, result.scores[0].shape[0], number_beams):
#         # lls = torch.tensor(0, dtype=torch.float)
#         # print(batch_num)
#         max_score = torch.tensor(-1*1e6, dtype=torch.float).to(device)
#         for beam_num in range(number_beams):
#             print([torch.max(result.scores[-1][batch_num+beam_num]), max_score])
#             max_score = torch.max(torch.stack([torch.max(result.scores[-1][batch_num+beam_num]), max_score]))
#         log_sent.append(max_score)
        
#     print(log_sent)
#     print(torch.stack(log_sent).sum())
#     print(torch.exp((-1*(torch.stack(log_sent).sum()))/result.sequences.shape[1]))

# if __name__ == "__main__":
#     main()

# def main(): 
#     model_checkpoint = 'facebook/bart-large-cnn' #'facebook/bart-base' #'a1noack/bart-large-gigaword'
#     tokenizer = BartTokenizerFast.from_pretrained(model_checkpoint)
#     model = BartForConditionalGeneration.from_pretrained(model_checkpoint, return_dict=True)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(device)
#     test = load_dataset("billsum", split='test[:15]')
#     print(test)
#     print(model.config.max_length)
#     encodings =  tokenizer(test['text'], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device)

#     model = model.to(device)
#     model.eval()
#     number_beams = 8
#     with torch.no_grad():
#         result = model.generate(encodings['input_ids'],  num_beams=number_beams, return_dict_in_generate=True, max_length=model.config.max_length, output_scores=True, output_attentions=True)
    
#     all = []
#     log_sent = []
#     print(result.sequences.shape)
#     print("Billsum vocab size: ", result.scores[0].shape[1])
#     print("Input ids size", encodings['input_ids'].shape)
#     for batch_num in range(0, result.scores[0].shape[0], number_beams):
#         # lls = torch.tensor(0, dtype=torch.float)
#         # print(batch_num)
#         max_score = torch.tensor(-1*1e6, dtype=torch.float).to(device)
#         for beam_num in range(number_beams):
#             print([torch.max(result.scores[-1][batch_num+beam_num]), max_score])
#             max_score = torch.max(torch.stack([torch.max(result.scores[-1][batch_num+beam_num]), max_score]))
#         log_sent.append(max_score)
        
#     print(log_sent)
#     print(torch.stack(log_sent).sum())
#     print(torch.exp((-1*(torch.stack(log_sent).sum()))/result.sequences.shape[1]))
#     # print(result.scores)
#     # print(result.scores[0].shape)
#     # print(max(result.scores[-1][-1]))
#     # print(max(result.scores[-2][-1]))
#     # print(max(result.scores[-3][-1]))
#     # print(max(result.scores[-4][-1]))
#     # print(result.sequences_scores[-1])
#     # print(result.sequences_scores[-1].sum())

#     # with torch.no_grad():
#     #     output = model(summary_ids, labels=summary_ids)

    

# if __name__ == "__main__":
#     main()

# def main():
#     model_checkpoint = 'facebook/bart-large-cnn' #'a1noack/bart-large-gigaword' #'facebook/bart-base'
#     tokenizer = BartTokenizerFast.from_pretrained(model_checkpoint)
#     model = BartForConditionalGeneration.from_pretrained(model_checkpoint, return_dict=True)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(device)
#     test = load_dataset("big_patent", "g", split='test[:16]')
#     print(test)
#     print(model.config.max_length)
#     encodings =  tokenizer(test['description'], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device)

#     model = model.to(device)
#     model.eval()
#     number_beams = 8
#     with torch.no_grad():
#         result = model.generate(encodings['input_ids'],  num_beams=number_beams, return_dict_in_generate=True, max_length=model.config.max_length, output_scores=True, output_attentions=True)
    
#     all = []
#     log_sent = []
#     print(result.sequences.shape)
#     print("big patent vocab size: ", result.scores[0].shape[1])
#     print("Input ids size", encodings['input_ids'].shape)
#     for batch_num in range(0, result.scores[0].shape[0], number_beams):
#         # lls = torch.tensor(0, dtype=torch.float)
#         # print(batch_num)
#         max_score = torch.tensor(-1*1e6, dtype=torch.float).to(device)
#         for beam_num in range(number_beams):
#             print([torch.max(result.scores[-1][batch_num+beam_num]), max_score])
#             max_score = torch.max(torch.stack([torch.max(result.scores[-1][batch_num+beam_num]), max_score]))
#         log_sent.append(max_score)
        
#     print(log_sent)
#     print(torch.stack(log_sent).sum())
#     print(torch.exp((-1*(torch.stack(log_sent).sum()))/result.sequences.shape[1]))

# if __name__ == "__main__":
#     main()

# Vanilla / Fine-tuned BART on XSum

def main():
    model_checkpoint = 'a1noack/bart-large-gigaword' # 'facebook/bart-large-cnn' #'facebook/bart-base'
    tokenizer = BartTokenizerFast.from_pretrained(model_checkpoint)
    model = BartForConditionalGeneration.from_pretrained(model_checkpoint, return_dict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    test = load_dataset("xsum", split='test[:20]')
    print(test)
    print(model.config.max_length)
    encodings =  tokenizer(test['text'], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device)

    model = model.to(device)
    model.eval()
    number_beams = 8
    with torch.no_grad():
        result = model.generate(encodings['input_ids'],  num_beams=number_beams, return_dict_in_generate=True, max_length=model.config.max_length, output_scores=True, output_attentions=True)
    
    all = []
    log_sent = []
    print(result.sequences.shape)
    print("XSum vocab size: ", result.scores[0].shape[1])
    print("Input ids size", encodings['input_ids'].shape)
    for batch_num in range(0, result.scores[0].shape[0], number_beams):
        # lls = torch.tensor(0, dtype=torch.float)
        # print(batch_num)
        max_score = torch.tensor(-1*1e6, dtype=torch.float).to(device)
        for beam_num in range(number_beams):
            print([torch.max(result.scores[-1][batch_num+beam_num]), max_score])
            max_score = torch.max(torch.stack([torch.max(result.scores[-1][batch_num+beam_num]), max_score]))
        log_sent.append(max_score)
        
    print(log_sent)
    print(torch.stack(log_sent).sum())
    print(torch.exp((-1*(torch.stack(log_sent).sum()))/result.sequences.shape[1]))

if __name__ == "__main__":
    main()

# Vanilla / Fine-tuned BART on CNN Dailymail

# def main():
#     model_checkpoint = 'a1noack/bart-large-gigaword' # 'facebook/bart-large-cnn' #'facebook/bart-base'
#     tokenizer = BartTokenizerFast.from_pretrained(model_checkpoint)
#     model = BartForConditionalGeneration.from_pretrained(model_checkpoint, return_dict=True)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(device)
#     test = load_dataset("cnn_dailymail", split='test[:20]')
#     print(test)
#     print(model.config.max_length)
#     encodings =  tokenizer(test['text'], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device)

#     model = model.to(device)
#     model.eval()
#     number_beams = 8
#     with torch.no_grad():
#         result = model.generate(encodings['input_ids'],  num_beams=number_beams, return_dict_in_generate=True, max_length=model.config.max_length, output_scores=True, output_attentions=True)
    
#     all = []
#     log_sent = []
#     print(result.sequences.shape)
#     print("CNN Dailymail vocab size: ", result.scores[0].shape[1])
#     print("Input ids size", encodings['input_ids'].shape)
#     for batch_num in range(0, result.scores[0].shape[0], number_beams):
#         # lls = torch.tensor(0, dtype=torch.float)
#         # print(batch_num)
#         max_score = torch.tensor(-1*1e6, dtype=torch.float).to(device)
#         for beam_num in range(number_beams):
#             print([torch.max(result.scores[-1][batch_num+beam_num]), max_score])
#             max_score = torch.max(torch.stack([torch.max(result.scores[-1][batch_num+beam_num]), max_score]))
#         log_sent.append(max_score)
        
#     print(log_sent)
#     print(torch.stack(log_sent).sum())
#     print(torch.exp((-1*(torch.stack(log_sent).sum()))/result.sequences.shape[1]))

# if __name__ == "__main__":
#     main()

