# only python 2.X
import os
import pickle

import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import evaluate


def prepare(data):
    users = {}
    items = {}
    data_list = []
    count_skipp = 0
    cnt = 0
    for _, row in data.iterrows():
        try:
            play_count = int(row[2])
        except ValueError:
            print('Value error:', row[2])
            play_count = 0
        if play_count <= 2:
            continue
        user = row[0]
        song = row[1]
        if song is None:
            count_skipp += 1
            continue
        user_i = users.get(user)
        item_i = items.get(song)
        if user_i is None:
            user_i = len(users)
            users[user] = len(users)
        if item_i is None:
            item_i = len(items)
            items[song] = len(items)
        data_list.append([user_i, item_i, play_count])
        cnt += 1
        print('Download {}%'.format(cnt/float(len(data))*100))

    new_data = pd.DataFrame(data_list)
    print(count_skipp)

    with open('group_index_dict.tit', 'wb') as f:
        pickle.dump(items, f)
    with open('users_index_dict.tit', 'wb') as f:
        pickle.dump(users, f)
    # with open('item_user_matrix.tit', 'wb') as f:
    #     pickle.dump(resMatrix, f)

    return new_data


def main():
    file_path = os.path.expanduser('res_data_with_keys.csv')
    data_ini = pd.read_csv('train_triplets.txt', sep=' ', header=None)
    data_ini.columns = ['user', 'song', 'play_count']
    data = data_ini
    data = data[data.play_count >= 3]
    df = data
    res_df = prepare(df)
    res_df.to_csv('res_data_with_keys.csv', header=None, index=False)

    with open('users_index_dict.tit', 'rb') as f:
        users_dict = dict(pickle.load(f))

    with open('group_index_dict.tit', 'rb') as f:
        items_dict = dict(pickle.load(f))

    print('User\'s dictionary of keys: {}'.format(list(i for i in users_dict.items() if i[1] < 100)))
    print('Song\'s dictionary of keys: {}'.format(list(i for i in items_dict.items() if i[1] < 100)))

    reader = Reader(line_format='user item rating', sep=',')
    data = Dataset.load_from_file(file_path, reader=reader)
    data.split(n_folds=5)

    algo = SVD()

    perf = evaluate(algo=algo,
                    data=data,
                    measures=['RMSE', 'MAE'],
                    with_dump=True,
                    # dump_dir='/media/mart/Data/OtherThings'
                    dump_dir='./')

    print perf