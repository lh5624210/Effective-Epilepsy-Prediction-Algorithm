import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from models import *
from tools import *

cut = 5
step = 5
epochs = 100
input_D = (1280, 18)
batch_size = 64
group = 2 * 60  # min
group_size = int(group / 5)

input_path = 'input2'
output_path = 'output_2min'
model_name = 'cnn-et'
set_seed(0)

folders = os.listdir(f'{input_path}')
res_total = {'ID': []}
for folder in folders:
    id = int(folder[3:])
    for seed in range(10):
        if os.path.exists(f'{output_path}/{model_name}.csv'):
            old_df = pd.read_csv(f'{output_path}/{model_name}.csv')
            if seed in old_df.loc[(old_df['ID'] == id), 'seed'].values:
                continue
        else:
            old_df = None

        set_seed(seed)
        interictal_files = os.listdir(f'{input_path}\{folder}\interictal')
        preictal_files = os.listdir(f'{input_path}\{folder}\preictal')
        if len(interictal_files) < 3 or len(preictal_files) < 3 or len(interictal_files) < len(preictal_files):
            continue

        res_person = {}
        for index_pi in range(len(preictal_files)):
            test_pi = pd.read_csv(f'{input_path}\{folder}\preictal\{preictal_files[index_pi]}')
            tmp_pi_name = preictal_files[index_pi]
            tmp_pi_index = index_pi
            preictal_files.pop(tmp_pi_index)

            index_ii = random.randrange(0, len(interictal_files))
            test_ii = pd.read_csv(f'{input_path}\{folder}\interictal\{interictal_files[index_ii]}')
            tmp_ii_index = index_ii
            tmp_ii_name = interictal_files[index_ii]
            interictal_files.pop(tmp_ii_index)

            test_lst = []
            lst = df2slicelist(test_ii, 0)
            test_lst.extend(lst)
            lst = df2slicelist(test_pi, 1)
            test_lst.extend(lst)

            train_all = []
            rd_index_lst = list(range(len(interictal_files)))
            random.shuffle(rd_index_lst)
            for i in range(len(preictal_files)):
                j = rd_index_lst[i]

                df_ii = pd.read_csv(f'{input_path}\{folder}\interictal\{interictal_files[j]}')
                lst = df2slicelist(df_ii, 0)
                train_all.extend(lst)

                df_pi = pd.read_csv(f'{input_path}\{folder}\preictal\{preictal_files[i]}')
                lst = df2slicelist(df_pi, 1)
                train_all.extend(lst)

            interictal_files.insert(index_ii, tmp_ii_name)
            preictal_files.insert(index_pi, tmp_pi_name)

            test_set = Data_set(test_lst)
            train_all_set = Data_set(train_all)
            train_set, val_set = random_split(train_all_set, [0.8, 0.2])

            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_set, batch_size=group_size, shuffle=False, num_workers=0)

            model = CNN__ET()
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters())
            model = train(train_loader, val_loader, epochs=epochs, model_name=model_name, model=model, loss_fn=loss_fn,
                          optimizer=optimizer, n_stop=8)

            res = test(test_loader, model=model, indicators=['Acc', 'Sen', 'Spe', 'FPR'], cut=cut, group=True,
                       threshold=0.6)
            for indicator in res:
                if indicator not in res_person:
                    res_person[indicator] = [res[indicator]]
                else:
                    res_person[indicator].append(res[indicator])
        print(res_person)

        for indicator in res_person:
            tmp = np.mean(res_person[indicator])
            if indicator not in res_total:
                res_total[indicator] = [tmp]
            else:
                res_total[indicator].append(tmp)

        res_total['ID'].append(id)
        if 'seed' not in res_total:
            res_total['seed'] = [seed]
        else:
            res_total['seed'].append(seed)
        print(res_total)

        df = pd.DataFrame(res_total)
        if old_df is None:
            df.to_csv(f'{output_path}/{model_name}.csv', index=False)
        else:
            df = df.iloc[-1, :]
            old_df.loc[len(old_df)] = df
            old_df.to_csv(f'{output_path}/{model_name}.csv', index=False)
