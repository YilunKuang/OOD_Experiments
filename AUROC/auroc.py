from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pickle

def compute_auroc(id_pps, ood_pps, normalize=False, return_curve=False):
    y = np.concatenate((np.ones_like(ood_pps), np.zeros_like(id_pps)))
    scores = np.concatenate((ood_pps, id_pps))
    if normalize:
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    if return_curve:
        return roc_curve(y, scores)
    else:
        return roc_auc_score(y, scores)

def main():
    dataset_lst = ['wikihow', 'gigaword', 'xsum', 'billsum']#, 'big_patent', 'cnn_dailymail']
    dataset_dir = ['Wikihow', 'Gigaword', 'XSum', 'Billsum']#, 'Big-Patent', 'CNN']

    for i in range(len(dataset_lst)):
        iid_dataset_name = 'wikihow'
        ind_iid_dataset_dir = dataset_lst.index(iid_dataset_name)
        iid_file_name = '/scratch/yk2516/OOD_Text_Generation/BART-'+dataset_dir[ind_iid_dataset_dir]+'/ppl_result/summary_ppl_'+ iid_dataset_name +'.pkl'

        ood_dataset_name = dataset_lst[i]
        ind_ood_dataset_dir = dataset_lst.index(ood_dataset_name)
        # ood_file_name = '/home/nm3571/summarization/results/fine_tuned_gpt2/summary_ppl_' + ood_dataset_name + '.pkl'
        ood_file_name = '/scratch/yk2516/OOD_Text_Generation/BART-'+dataset_dir[ind_ood_dataset_dir]+'/ppl_result/summary_ppl_'+ood_dataset_name+'.pkl'

        iid_ppls = []
        ood_ppls = []
        with open(iid_file_name, 'rb') as f:
            iid_ppls = pickle.load(f)
        with open(ood_file_name, 'rb') as g:
            ood_ppls = pickle.load(g)

        # print(ood_ppls)

        # Uncomment below if you stored tensors in the above file (which is the case for BART).
        for idx, tens in enumerate(iid_ppls):
           iid_ppls[idx] = tens.cpu().detach().item()
        for idx, tens in enumerate(ood_ppls):
           ood_ppls[idx] = tens.cpu().detach().item()

        # print("after", iid_ppls)
        result = compute_auroc(iid_ppls, ood_ppls)
        print("AUROC for " + iid_dataset_name + " vs " + ood_dataset_name)
        print(result)

if __name__ == "__main__":
    main()
    
