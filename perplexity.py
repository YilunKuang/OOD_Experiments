from transformers import AutoTokenizer, AutoModel, BartForConditionalGeneration, BartTokenizerFast
from datasets import load_dataset
import torch as torch
from tqdm import tqdm
from datasets import load_metric

def main():
    rouge_metric = load_metric('rouge')
    model_checkpoint = 'a1noack/bart-large-gigaword'
    tokenizer = BartTokenizerFast.from_pretrained("a1noack/bart-large-gigaword")
    model = BartForConditionalGeneration.from_pretrained(model_checkpoint, return_dict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    test = load_dataset("gigaword", split='test[:30]')
    print(test)
    print(model.config.max_length)
    encodings =  tokenizer(test['document'], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device)

    model = model.to(device)
    model.eval()
    result = model.generate(encodings['input_ids'],  num_beams=1, return_dict_in_generate=True, max_length=model.config.max_length, output_scores=True, output_attentions=True)
    
    all = []
    log_sent = []
    print(result.sequences.shape)
    for batch_num in range(result.scores[0].shape[0]):
        lls = torch.tensor(0, dtype=torch.float).to(device)
        # print(batch_num)
        for i in range(len(result.scores)): 
            # print(i)
            a = torch.softmax(result.scores[i][batch_num], dim=0)
            a = torch.log(torch.max(a))
            lls = lls + a
        log_sent.append(lls)
        lls = (-1*lls)/result.sequences[batch_num].shape[0]
        lls = torch.exp(lls)
        all.append(lls)
    print(all)
    print(log_sent)
    print(torch.stack(log_sent).sum())
    print(torch.exp((-1*(torch.stack(log_sent).sum()))/result.sequences.shape[1]))
    # print(result.scores)
    # print(result.scores[0].shape)
    # print(max(result.scores[-1][-1]))
    # print(max(result.scores[-2][-1]))
    # print(max(result.scores[-3][-1]))
    # print(max(result.scores[-4][-1]))
    # print(result.sequences_scores[-1])
    # print(result.sequences_scores[-1].sum())

    # with torch.no_grad():
    #     output = model(summary_ids, labels=summary_ids)

    

if __name__ == "__main__":
    main()


# def main():
#     rouge_metric = load_metric('rouge')
#     model_checkpoint = 'a1noack/bart-large-gigaword'
#     tokenizer = BartTokenizerFast.from_pretrained("a1noack/bart-large-gigaword")
#     model = BartForConditionalGeneration.from_pretrained(model_checkpoint, return_dict=True)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(device)
#     test = load_dataset("wikihow", "all", data_dir='/scratch/nm3571', split='test[:30]')
#     print(test)
#     print(model.config.max_length)
#     encodings =  tokenizer(test['text'], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device)

#     model = model.to(device)
#     model.eval()
#     result = model.generate(encodings['input_ids'],  num_beams=1, return_dict_in_generate=True, max_length=128, output_scores=True, output_attentions=True)

#     all = []
#     log_sent = []
#     print(result.sequences.shape)
#     for batch_num in range(result.scores[0].shape[0]):
#         lls = torch.tensor(0, dtype=torch.float)
#         # print(batch_num)
#         for i in range(len(result.scores)): 
#             # print(i)
#             a = torch.softmax(result.scores[i][batch_num], dim=0)
#             # print(a.shape)
#             a = torch.log(torch.max(a))
#             lls = lls + a
#         log_sent.append(lls)
#         lls = (-1*lls)/result.sequences[batch_num].shape[0]
#         lls = torch.exp(lls)
#         all.append(lls)
#     print(all)
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
