import os
import mne
import re
import time
import datetime
import pandas as pd
import torch


def find_seizures(data_path):
    folders = os.listdir(data_path)
    total_seizures = {}
    for folder in folders:
        if folder[:3] == 'chb':
            path = folder
            files = os.listdir(f'{data_path}\{path}')
            seizures = {}
            for file in files:
                if file[-3:] == 'txt':
                    with open(f'{data_path}\{path}\{file}', 'r', encoding='utf-8') as txt:
                        find = [0, 0]
                        name = None
                        for line in txt.readlines():
                            line = line.strip('\n')
                            file_name = re.findall(r'(?<=File Name: ).*$', line)
                            if file_name:
                                name = file_name[0]
                            else:
                                find_seconds = re.findall(r'(?<=:).* (?=seconds)', line)
                                if find_seconds:
                                    seconds = int(find_seconds[0])
                                    if seconds == 0 or find[0] == find[1] == 0:
                                        find[0] = seconds
                                    else:
                                        find[1] = seconds
                                        if name not in seizures:
                                            seizures[name] = [tuple(find)]
                                        else:
                                            seizures[name].append(tuple(find))
                                        find = [0, 0]
            total_seizures[path] = seizures
    return total_seizures


def search_preictal_and_interictal(input_path, total_seizures, interval=4.0 * 3600, preictal=1.0 * 3600):
    folders = os.listdir(input_path)
    total_tag = {}
    for folder in folders:
        if folder[:3] == 'chb':
            path = folder
            seizures = total_seizures[path]
            lst = list(seizures.keys())
            files = os.listdir(f'{input_path}\{path}')
            new = []
            for file in files:
                if file[-3:] == 'edf':
                    new.append(file)
            files = new
            files_time = []
            tag = {}

            old_end = None
            old_i = None
            tail_doing = False
            tail_add = 0
            for i in range(len(files)):
                file = files[i]
                raw = mne.io.read_raw_edf(f'{input_path}\{path}\{file}')
                data, times = raw.get_data(return_times=True)
                files_time.append(int(f'{times[-1]:.0f}'))
                tag[file] = {'preictal': [], 'interictal': []}
                if file in seizures:
                    records = seizures[file]
                    for j in range(len(records)):
                        record = records[j]
                        start, end = record
                        if old_end:
                            cum = 0
                            for k in range(old_i, i):
                                if k == old_i:
                                    cum += files_time[k] - old_end
                                else:
                                    cum += files_time[k]
                            cum += start

                            add = 0
                            if cum < preictal:
                                pass
                            elif preictal <= cum < 2 * interval + preictal:  # 只有发作前期
                                preictal_done = False
                                for k in range(i, old_i - 1, -1):
                                    if k == i:
                                        add += start
                                    else:
                                        add += files_time[k]

                                    if add < preictal:
                                        if add < files_time[k]:
                                            tag[files[k]]['preictal'].append((0, add))
                                        else:
                                            tag[files[k]]['preictal'].append((0, files_time[k]))
                                    else:
                                        if add < files_time[k]:
                                            tag[files[k]]['preictal'].append((add - preictal, add))
                                        else:
                                            tag[files[k]]['preictal'].append((add - preictal, files_time[k]))
                                        preictal_done = True

                                    if preictal_done:
                                        break

                            elif 2 * interval + preictal <= cum:  # 都存在
                                preictal_done = False
                                interictal_doing = False
                                interictal_done = False
                                for k in range(i, old_i - 1, -1):
                                    if k == i:
                                        add += start
                                    else:
                                        add += files_time[k]

                                    if add < preictal:
                                        if add < files_time[k]:
                                            tag[files[k]]['preictal'].append((0, add))
                                        else:
                                            tag[files[k]]['preictal'].append((0, files_time[k]))

                                    elif preictal <= add < interval:
                                        if not preictal_done:
                                            if add < files_time[k]:
                                                tag[files[k]]['preictal'].append((add - preictal, add))
                                            else:
                                                tag[files[k]]['preictal'].append((add - preictal, files_time[k]))
                                            preictal_done = True

                                    elif interval <= add < cum - interval:
                                        if not preictal_done:
                                            if add < files_time[k]:
                                                tag[files[k]]['preictal'].append((add - preictal, add))
                                            else:
                                                tag[files[k]]['preictal'].append((add - preictal, files_time[k]))
                                            preictal_done = True

                                        if interictal_doing:
                                            tag[files[k]]['interictal'].append((0, files_time[k]))
                                        else:
                                            tag[files[k]]['interictal'].append((0, add - interval))
                                            interictal_doing = True

                                    elif cum - interval <= add:
                                        if not preictal_done:
                                            if add < files_time[k]:
                                                tag[files[k]]['preictal'].append((add - preictal, add))
                                            else:
                                                tag[files[k]]['preictal'].append((add - preictal, files_time[k]))
                                            preictal_done = True
                                        if not interictal_done:
                                            if interictal_doing:
                                                tag[files[k]]['interictal'].append(
                                                    (add - cum + interval, files_time[k]))
                                            else:
                                                tag[files[k]]['interictal'].append(
                                                    (add - cum + interval, add - interval))
                                                interictal_doing = True
                                            interictal_done = True
                                        if preictal_done and interictal_done:
                                            break

                        else:
                            cum = 0
                            for k in range(i):
                                cum += files_time[k]
                            cum += start

                            add = 0
                            if cum < preictal:
                                pass
                            elif preictal <= cum < interval + preictal:  # 只存在发作前期
                                preictal_done = False
                                for k in range(i, -1, -1):
                                    if k == i:
                                        add += start
                                    else:
                                        add += files_time[k]

                                    if add < preictal:
                                        if add < files_time[k]:
                                            tag[files[k]]['preictal'].append((0, add))
                                        else:
                                            tag[files[k]]['preictal'].append((0, files_time[k]))
                                    else:
                                        if add < files_time[k]:
                                            tag[files[k]]['preictal'].append((add - preictal, add))
                                        else:
                                            tag[files[k]]['preictal'].append((add - preictal, files_time[k]))
                                        preictal_done = True

                                    if preictal_done:
                                        break
                            elif interval + preictal <= cum:  # 都存在
                                preictal_done = False
                                interictal_doing = False
                                for k in range(i, -1, -1):
                                    if k == i:
                                        add += start
                                    else:
                                        add += files_time[k]

                                    if add < preictal:
                                        if add < files_time[k]:
                                            tag[files[k]]['preictal'].append((0, add))
                                        else:
                                            tag[files[k]]['preictal'].append((0, files_time[k]))

                                    elif preictal <= add < interval:
                                        if not preictal_done:
                                            if add < files_time[k]:
                                                tag[files[k]]['preictal'].append((add - preictal, add))
                                            else:
                                                tag[files[k]]['preictal'].append((add - preictal, files_time[k]))
                                            preictal_done = True

                                    elif interval <= add:
                                        if not preictal_done:
                                            if add < files_time[k]:
                                                tag[files[k]]['preictal'].append((add - preictal, add))
                                            else:
                                                tag[files[k]]['preictal'].append((add - preictal, files_time[k]))
                                            preictal_done = True
                                        if interictal_doing:
                                            tag[files[k]]['interictal'].append((0, files_time[k]))
                                        else:
                                            tag[files[k]]['interictal'].append((0, add - interval))
                                            interictal_doing = True

                        old_end = end
                        old_i = i

                    if file == lst[-1]:
                        tail_add = files_time[old_i] - old_end
                        if interval <= tail_add:
                            tag[files[i]]['interictal'].append((old_end + interval, files_time[i]))
                            tail_doing = True

                else:
                    if tail_doing:
                        tag[files[i]]['interictal'].append((0, files_time[i]))
                        continue
                    if old_i and files[old_i] == lst[-1]:
                        tail_add += files_time[i]
                        if interval <= tail_add:
                            tag[files[i]]['interictal'].append((files_time[i] - tail_add + interval, files_time[i]))
                            tail_doing = True

            total_tag[path] = tag

    return total_tag


def edf2csv(data_path, output_path, total_tag, channels, total_seizures=None):
    folders = os.listdir(data_path)
    for folder in folders:
        if folder[:3] == 'chb':
            path = folder
            files = os.listdir(f'{data_path}\{path}')
            seizures = None
            if total_seizures:
                seizures = total_seizures[path]
            tag = total_tag[path]
            for file in files:
                if file[-3:] == 'edf':
                    raw = mne.io.read_raw_edf(f'{data_path}\{path}\{file}')
                    raw.load_data()
                    original_raw = raw.copy()
                    ch_names = original_raw.info['ch_names']
                    if 'T8-P8-0' in ch_names:
                        rename_dict = {'T8-P8-0': 'T8-P8'}
                        original_raw.rename_channels(rename_dict)
                    try:
                        original_raw.pick_channels(channels, ordered=True)
                    except:
                        continue

                    df = original_raw.to_data_frame()
                    df['label'] = None

                    # if total_seizures:
                    #     for key in seizures:
                    #         if key == file:
                    #             def func(x, lst: list[tuple]):
                    #                 for start, end in lst:
                    #                     if start <= x < end:
                    #                         return 2
                    #                 else:
                    #                     return None
                    #
                    #             df['label'] = df['time'].apply(lambda x: func(x, lst=seizures[key]))
                    #             break
                    #     else:
                    #         df['label'] = None

                    preictal_lst = tag[file]['preictal']
                    interictal_lst = tag[file]['interictal']
                    for pre_start, pre_end in preictal_lst:
                        df.loc[(pre_start <= df['time']) & (df['time'] < pre_end), 'label'] = 1
                    for inter_start, inter_end in interictal_lst:
                        df.loc[(inter_start <= df['time']) & (df['time'] < inter_end), 'label'] = 0

                    df.dropna(axis=0, inplace=True)
                    if len(df) >= 1:
                        if not os.path.exists(f'{output_path}\{path}'):
                            os.mkdir(f'{output_path}\{path}')
                        df.to_csv(f'{output_path}\{path}\{file[:-4]}.csv', index=False)


def cut_data_from_csv(in_path, out_path, cut=5, step=5, preictal=1.0 * 3600, save_tail=False):
    global file
    folders = os.listdir(f'{in_path}')
    for folder in folders:
        files = os.listdir(f'{in_path}\{folder}')
        interictal_lst = []
        preictal_lst = []
        interictal_file_num = 1
        preictal_file_num = 1
        if not os.path.exists(f'{out_path}\{folder}'):
            os.mkdir(f'{out_path}\{folder}')
        if not os.path.exists(f'{out_path}\{folder}\interictal'):
            os.mkdir(f'{out_path}\{folder}\interictal')
        if not os.path.exists(f'{out_path}\{folder}\preictal'):
            os.mkdir(f'{out_path}\{folder}\preictal')
        res = None
        res_label = None
        for file in files:
            df = pd.read_csv(f'{in_path}\{folder}\{file}')
            start = 0
            if not res_label:
                end = 256 * cut
            else:
                end = 256 * cut - len(res)

            while end <= len(df):
                data = df.iloc[start:end, :]
                start_label = data.iloc[0, -1]
                end_label = data.iloc[-1, -1]
                if res_label:
                    if res_label == start_label:
                        if start_label == end_label:
                            data = pd.concat([res, data], axis=0)
                            data = data.drop(['time', 'label'], axis=1)
                            if end_label == 0:
                                interictal_lst.append(data)
                            else:
                                preictal_lst.append(data)

                            if len(preictal_lst) == preictal / cut:
                                tmp = pd.concat(preictal_lst, axis=0)
                                tmp.to_csv(f'{out_path}\{folder}\preictal\{file[:-4]}_{preictal_file_num}.csv',
                                           index=False)
                                preictal_file_num += 1
                                preictal_lst = []
                            if len(interictal_lst) == preictal / cut:
                                tmp = pd.concat(interictal_lst, axis=0)
                                tmp.to_csv(f'{out_path}\{folder}\interictal\{file[:-4]}_{interictal_file_num}.csv',
                                           index=False)
                                interictal_file_num += 1
                                interictal_lst = []

                            res = None
                            res_label = None
                            start = end
                            end = start + 256 * cut
                        else:
                            res = None
                            res_label = None
                            start = data[data['label'] == end_label].index[0]
                            end = start + 256 * cut

                    else:
                        res = None
                        res_label = None
                        start = 0
                        end = 256 * cut
                    continue
                else:
                    if start_label == end_label:
                        data = data.drop(['time', 'label'], axis=1)
                        if end_label == 0:
                            interictal_lst.append(data)
                        else:
                            preictal_lst.append(data)

                        if len(preictal_lst) == preictal / cut:
                            tmp = pd.concat(preictal_lst, axis=0)
                            tmp.to_csv(f'{out_path}\{folder}\preictal\{file[:-4]}_{preictal_file_num}.csv',
                                       index=False)
                            preictal_file_num += 1
                            preictal_lst = []
                        if len(interictal_lst) == preictal / cut:
                            tmp = pd.concat(interictal_lst, axis=0)
                            tmp.to_csv(f'{out_path}\{folder}\interictal\{file[:-4]}_{interictal_file_num}.csv',
                                       index=False)
                            interictal_file_num += 1
                            interictal_lst = []

                    else:
                        start = data[data['label'] == end_label].index[0]
                        end = start + 256 * cut
                        continue
                start += 256 * step
                end += 256 * step
            else:
                data = df.iloc[start:, :]
                if len(data) == 0:
                    continue
                start_label = data.iloc[0, -1]
                end_label = data.iloc[-1, -1]
                if start_label == end_label:
                    if res_label and res_label == start_label:
                        res = pd.concat([res, data], axis=0)
                    else:
                        res = data
                        res_label = end_label
                else:
                    start = data[data['label'] == end_label].index[0]
                    res = data[start:, :]
                    res_label = end_label

        if save_tail:
            if interictal_lst:
                df = pd.concat(interictal_lst, axis=0)
                df.to_csv(f'{out_path}\{folder}\interictal\{file[:-4]}_{interictal_file_num}.csv', index=False)
            if preictal_lst:
                df = pd.concat(preictal_lst, axis=0)
                df.to_csv(f'{out_path}\{folder}\preictal\{file[:-4]}_{preictal_file_num}.csv', index=False)


def cut_data_order_from_csv(in_path, out_path, cut=5, step=5, preictal=1.0 * 3600, save_tail=False):
    global file
    folders = os.listdir(f'{in_path}')
    for folder in folders:
        files = os.listdir(f'{in_path}\{folder}')
        interictal_lst = []
        preictal_lst = []
        interictal_file_num = 1
        preictal_file_num = 1
        if not os.path.exists(f'{out_path}\{folder}'):
            os.mkdir(f'{out_path}\{folder}')

        res = None
        res_label = None
        for file in files:
            df = pd.read_csv(f'{in_path}\{folder}\{file}')
            start = 0
            if not res_label:
                end = 256 * cut
            else:
                end = 256 * cut - len(res)

            while end <= len(df):
                data = df.iloc[start:end, :]
                start_label = data.iloc[0, -1]
                end_label = data.iloc[-1, -1]
                if res_label:
                    if res_label == start_label:
                        if start_label == end_label:
                            data = pd.concat([res, data], axis=0)
                            data = data.drop(['time', 'label'], axis=1)
                            if end_label == 0:
                                interictal_lst.append(data)
                            else:
                                preictal_lst.append(data)

                            if len(preictal_lst) == preictal / cut:
                                tmp = pd.concat(preictal_lst, axis=0)
                                tmp.to_csv(f'{out_path}\{folder}\{file[:-4]}_{preictal_file_num}_preictal.csv',
                                           index=False)
                                preictal_file_num += 1
                                preictal_lst = []
                            if len(interictal_lst) == preictal / cut:
                                tmp = pd.concat(interictal_lst, axis=0)
                                tmp.to_csv(f'{out_path}\{folder}\{file[:-4]}_{interictal_file_num}_interictal.csv',
                                           index=False)
                                interictal_file_num += 1
                                interictal_lst = []

                            res = None
                            res_label = None
                            start = end
                            end = start + 256 * cut
                        else:
                            res = None
                            res_label = None
                            start = data[data['label'] == end_label].index[0]
                            end = start + 256 * cut

                    else:
                        res = None
                        res_label = None
                        start = 0
                        end = 256 * cut
                    continue
                else:
                    if start_label == end_label:
                        data = data.drop(['time', 'label'], axis=1)
                        if end_label == 0:
                            interictal_lst.append(data)
                        else:
                            preictal_lst.append(data)

                        if len(preictal_lst) == preictal / cut:
                            tmp = pd.concat(preictal_lst, axis=0)
                            tmp.to_csv(f'{out_path}\{folder}\{file[:-4]}_{preictal_file_num}_preictal.csv',
                                       index=False)
                            preictal_file_num += 1
                            preictal_lst = []
                        if len(interictal_lst) == preictal / cut:
                            tmp = pd.concat(interictal_lst, axis=0)
                            tmp.to_csv(f'{out_path}\{folder}\{file[:-4]}_{interictal_file_num}_interictal.csv',
                                       index=False)
                            interictal_file_num += 1
                            interictal_lst = []

                    else:
                        start = data[data['label'] == end_label].index[0]
                        end = start + 256 * cut
                        continue
                start += 256 * step
                end += 256 * step
            else:
                data = df.iloc[start:, :]
                if len(data) == 0:
                    continue
                start_label = data.iloc[0, -1]
                end_label = data.iloc[-1, -1]
                if start_label == end_label:
                    if res_label and res_label == start_label:
                        res = pd.concat([res, data], axis=0)
                    else:
                        res = data
                        res_label = end_label
                else:
                    start = data[data['label'] == end_label].index[0]
                    res = data[start:, :]
                    res_label = end_label

        if save_tail:
            if interictal_lst:
                df = pd.concat(interictal_lst, axis=0)
                df.to_csv(f'{out_path}\{folder}\{file[:-4]}_{interictal_file_num}_interictal.csv', index=False)
            if preictal_lst:
                df = pd.concat(preictal_lst, axis=0)
                df.to_csv(f'{out_path}\{folder}\{file[:-4]}_{preictal_file_num}_preictal.csv', index=False)


def seizures_num_and_time(total_seizures):
    res = {'name':[],'num':[],'time':[]}
    for folder in total_seizures:
        seizures = total_seizures[folder]
        num = 0
        time = 0
        for file in seizures:
            for start, end in seizures[file]:
                time += end - start
                num += 1
        res['name'].append(folder)
        res['num'].append(num)
        res['time'].append(time)
    return res


if __name__ == '__main__':
    channels = ['P3-O1', 'F7-T7', 'F4-C4', 'FP1-F7', 'T7-P7', 'P7-O1', 'FZ-CZ', 'F3-C3', 'FP2-F8', 'P8-O2', 'FP2-F4',
                'P4-O2', 'FP1-F3', 'CZ-PZ', 'C4-P4', 'F8-T8', 'T8-P8', 'C3-P3']
    data_path = 'chb-mit-scalp-eeg-database-1.0.0'
    output_path = 'input'
    interval = 4.0 * 3600
    preictal = 1.0 * 3600
    cut = 5
    step = 5
    total_seizures = find_seizures(data_path)
    # res = seizures_num_and_time(total_seizures)
    # res = pd.DataFrame(res)
    # res.to_csv('发作次数和时间.csv', index=False)

    total_tag = search_preictal_and_interictal(data_path, total_seizures,interval=interval,preictal=preictal)
    edf2csv(data_path, output_path, total_tag, channels)
    cut_data_from_csv(output_path, out_path='input', cut=cut, step=step, preictal=preictal)
