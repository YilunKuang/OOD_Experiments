from transformers import AutoTokenizer, AutoModel, BartForConditionalGeneration, BartTokenizerFast
from datasets import load_dataset
import torch as torch
from tqdm import tqdm
from datasets import load_metric

def main():
    rouge_metric = load_metric('rouge')
    model_checkpoint = 'a1noack/bart-large-gigaword'
    tokenizer = BartTokenizerFast.from_pretrained("a1noack/bart-large-gigaword")
    model = BartForConditionalGeneration.from_pretrained(model_checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    test = load_dataset("gigaword", split='test[:30]')
    print(test)
    print(model.config.max_length)
    encodings =  tokenizer(test['document'], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device)

    model = model.to(device)
    summary_ids = model.generate(encodings['input_ids'],  max_length=model.config.max_length, early_stopping=True)
    print("done")
    abc = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    print(abc)
    print(rouge_metric.compute(predictcions=abc,references=test["summary"]))

if __name__ == "__main__":
    main()

# from transformers import AutoTokenizer, AutoModel, BartForConditionalGeneration, BartTokenizerFast
# from datasets import load_dataset
# import torch as torch
# from tqdm import tqdm
# from datasets import load_metric

# def main():
#     rouge_metric = load_metric('rouge')
#     model_checkpoint = 'a1noack/bart-large-gigaword'
#     tokenizer = BartTokenizerFast.from_pretrained("a1noack/bart-large-gigaword")
#     model = BartForConditionalGeneration.from_pretrained(model_checkpoint)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(device)
#     test = load_dataset("wikihow", "all", data_dir="/scratch/nm3571", split='test[:30]')
#     print(test)
#     print(model.config.max_length)
#     encodings =  tokenizer(test['text'], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device)
#     model = model.to(device)
#     summary_ids = model.generate(encodings['input_ids'],  max_length=model.config.max_length, early_stopping=True)
#     abc = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
#     print(abc)
#     print(rouge_metric.compute(predictions=abc,references=test["headline"]))

# if __name__ == "__main__":
#     main()
    