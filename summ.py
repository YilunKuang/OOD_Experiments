
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import torch, tqdm

def main():
    model_checkpoint = 'a1noack/bart-large-gigaword'
    tokenizer = AutoTokenizer.from_pretrained("a1noack/bart-large-gigaword")
    model = AutoModel.from_pretrained(model_checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    test = load_dataset("wikihow", "all", data_dir="/scratch/nm3571", split='test[:30]')
    print(test)
    encodings =  tokenizer(test['text'], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device)
    max_length = model.config.max_length
    stride = 512

    lls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    print("PPL with stride", stride)
    print("is: ", ppl)


if __name__ == "__main__":
    main()