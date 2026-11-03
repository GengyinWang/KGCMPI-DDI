import os
import copy
import math
import numpy as np


class KFold(object):
    def __init__(self, n_splits=5, shuffle=True, random_state=42, condition='S1'):
        # np.random.seed(random_state)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.condition = condition
        self.random_state = random_state

    def split(self, data):
        rng = np.random.RandomState(self.random_state)
        folds_data = [None] * self.n_splits
        if self.condition == 'S1':
            print(f"Start splitting for {self.condition} (Stratified by Relation)...")
            relations = np.unique(data[:, 2])
            relation_data = {r: data[data[:, 2] == r] for r in relations}
            # max_relation_id = data[:, 2].max()
            # split_class_data = [[] for _ in range(max_relation_id + 1)]
            # for item in data:
            #     split_class_data[item[-1]].append(item)
            

            # split_class_data = [np.array(c_data) for c_data in split_class_data if len(c_data) > 0]

            fold_train_lists = [[] for _ in range(self.n_splits)]
            fold_test_lists = [[] for _ in range(self.n_splits)]
            for r, class_data in relation_data.items():
                num_samples = len(class_data)
                indices = np.arange(num_samples)
                if self.shuffle:
                    # np.random.shuffle(class_data)
                    rng.shuffle(indices)
                
                # num_samples = len(class_data)
                split_indices = np.array_split(indices, self.n_splits)
                
                for fold_idx in range(self.n_splits):
                    # test_indices = np.arange(i, num_samples, self.n_splits)
                    test_idx = split_indices[fold_idx]
                    train_idx = np.concatenate([split_indices[i] for i in range(self.n_splits) if i != fold_idx])
                    # mask = np.zeros(num_samples, dtype=bool)
                    # mask[test_indices] = True
                    
                    c_test = class_data[test_idx]
                    c_train = class_data[train_idx]
                    
                    if len(c_train) > 0:
                        fold_train_lists[fold_idx].append(c_train)
                    if len(c_test) > 0:
                        fold_test_lists[fold_idx].append(c_test)

            for fold_idx in range(self.n_splits):
                train_final = np.concatenate(fold_train_lists[fold_idx], axis=0) if len(fold_train_lists[fold_idx]) > 0 else np.empty((0, 3))
                test_final = np.concatenate(fold_test_lists[fold_idx], axis=0) if len(fold_test_lists[fold_idx]) > 0 else np.empty((0, 3))
                folds_data[fold_idx] = [train_final, test_final]
        else:
            print(f"Start splitting for {self.condition} (Split by Drug Nodes)...")
            # max_drug_id = int(data[:, :2].max())
            # all_drug_id = np.arange(0, max_drug_id + 1)
            all_drug_id = np.unique(data[:, :2])
            
            if self.shuffle:
                # np.random.shuffle(all_drug_id)
                rng.shuffle(all_drug_id)

            unknown_drug_num = math.floor(len(all_drug_id) / self.n_splits)

            for split_index in range(self.n_splits):
                start = split_index * unknown_drug_num
                end = (split_index + 1) * unknown_drug_num
                if split_index == self.n_splits - 1:
                    end = len(all_drug_id)
                
                unknown_drug_set = set(all_drug_id[start:end])
                
                train_data = []
                test_data = []

                for item in data:
                    h, t, r = item
                    h_is_unknown = h in unknown_drug_set
                    t_is_unknown = t in unknown_drug_set

                    if (not h_is_unknown) and (not t_is_unknown):
                        train_data.append(item)
                    
                    elif self.condition == 'S2' and (h_is_unknown != t_is_unknown):
                        test_data.append(item)
                    
                    elif self.condition == 'S3' and (h_is_unknown and t_is_unknown):
                        test_data.append(item)
                
                train_data = np.array(train_data) if len(train_data) > 0 else np.empty((0, 3))
                test_data = np.array(test_data) if len(test_data) > 0 else np.empty((0, 3))
                
                folds_data[split_index] = [train_data, test_data]

        return folds_data
