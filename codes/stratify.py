import pandas as pd 
import numpy as np 
from tqdm import tqdm

def stratify_df(df, new_col_name, n_folds=10, nr_clean_folds=0):
    # compute qualities as described in PTB-XL report
    quals = []
    for i, row in df.iterrows():
        q = 0
        if 'validated_by_human' in df.columns:
            if row.validated_by_human == True:
                q = 1
        quals.append(q)
    df['quality'] = quals

    # create stratified folds according to patients
    pat_ids = np.array(sorted(list(set(df.patient_id.values))))
    plabels = []
    pquals = []
    ecgs_per_patient = []

    for pid in tqdm(pat_ids):
        sel = df[df.patient_id == pid]
        l = np.concatenate([list(d.keys())for d in sel.scp_codes.values])
        if sel.sex.values[0] == 0:
            gender='male'
        else:
            gender='female'
        l = np.concatenate((l, [gender]*len(sel)))
        for age in sel.age.values:
            if age < 20:
                l = np.concatenate((l,['<20']))
            elif age >= 20 and age < 40:
                l = np.concatenate((l,['20-40']))
            elif age >= 40 and age < 60:
                l = np.concatenate((l,['40-60']))
            elif age >= 60 and age < 80:
                l = np.concatenate((l,['60-80']))
            elif age >= 80:
                l = np.concatenate((l,['>=80']))
        plabels.append(l)
        ecgs_per_patient.append(len(sel))
        pquals.append(sel.quality.min())
    classes = sorted(list(set([item for sublist in plabels for item in sublist])))

    stratified_data_ids, stratified_data = stratify(plabels, classes, [1/n_folds]*n_folds, pquals, ecgs_per_patient, nr_clean_folds)

    df[new_col_name] = np.zeros(len(df)).astype(int)
    for fold_i, fold_ids in tqdm(enumerate(stratified_data_ids)):
        ipat_ids = [pat_ids[pid] for pid in fold_ids]
        df[new_col_name][df.patient_id.isin(ipat_ids)] = fold_i + 1

    return df

def stratify(data, classes, ratios, qualities, ecgs_per_patient, nr_clean_folds=1):
   
    np.random.seed(0)

    #size is the number of ecgs
    size = np.sum(ecgs_per_patient)

    per_label_data = {c: [] for c in classes}
    for i, d in enumerate(data):
        for l in d:
            per_label_data[l].append(i)

    subset_sizes = [r * size for r in ratios]
    per_label_subset_sizes = { c: [r * len(per_label_data[c]) for r in ratios] for c in classes }

    stratified_data_ids = [set() for _ in range(len(ratios))]

    # For each sample in the data set
    print("Assigning patients to folds...")
    size_prev=size+1
    while size > 0:
        if(int(size_prev/1000) > int(size/1000)):
            print("Remaining patients/ecgs to distribute:",size,"non-empty labels:", np.sum([1 for l, label_data in per_label_data.items() if len(label_data)>0]))
        size_prev=size
        lengths = {
            l: len(label_data)
            for l, label_data in per_label_data.items()
        }
        try:

            label = min({k: v for k, v in lengths.items() if v > 0}, key=lengths.get)
        except ValueError:
            break
        unique_samples, unique_counts = np.unique(per_label_data[label],return_counts=True)
        idxs_sorted = np.argsort(unique_counts, kind='stable')[::-1]
        unique_samples = unique_samples[idxs_sorted]
        unique_counts =  unique_counts[idxs_sorted]
        
        # loop through all patient ids with this label
        for current_id, current_count in zip(unique_samples,unique_counts):
            
            subset_sizes_for_label = per_label_subset_sizes[label]

            if(qualities[current_id] < 1):
                subset_sizes_for_label = subset_sizes_for_label[:len(ratios)-nr_clean_folds]

            largest_subsets = np.argwhere(subset_sizes_for_label == np.amax(subset_sizes_for_label)).flatten()
            
            if len(largest_subsets) == 1:
                subset = largest_subsets[0]

            else:
                largest_subsets2 = np.argwhere(np.array(subset_sizes)[largest_subsets] == np.amax(np.array(subset_sizes)[largest_subsets])).flatten()
                subset = largest_subsets[np.random.choice(largest_subsets2)]

            stratified_data_ids[subset].add(current_id)

            size -= ecgs_per_patient[current_id]
            subset_sizes[subset] -= ecgs_per_patient[current_id]

            for l in data[current_id]:
                per_label_subset_sizes[l][subset] -= 1
               
            # Remove the sample from the dataset, meaning from all per_label dataset created
            for x in per_label_data.keys():
                per_label_data[x] = [y for y in per_label_data[x] if y!=current_id]
              
    # Create the stratified dataset as a list of subsets, each containing the orginal labels
    stratified_data_ids = [sorted(strat) for strat in stratified_data_ids]
    stratified_data = [
        [data[i] for i in strat] for strat in stratified_data_ids
    ]

    return stratified_data_ids, stratified_data

