# -*- coding: utf-8 -*-
import pickle

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import spdiags
from sklearn.preprocessing import normalize

TRIPLETS_NUMBER = 10000000


def transform_first_data():
    with open('train_triplets.txt', 'r') as f:
        with open('train_triplets_sub.csv', 'w') as g:
            for i in range(TRIPLETS_NUMBER):
                g.write(f.readline().replace('	', ','))


def download_and_convert_data(data):
    users = {}
    items = {}
    res_matrix = sparse.lil_matrix((len(data['user'].unique()), len(data['song'].unique())))
    count_skipp = 0
    cnt = 0
    for _, row in data.iterrows():
        try:
            play_count = int(row['play_count'])
        except ValueError:
            print("Value error:", row['play_count'])
            play_count = 0
        if play_count <= 2:
            continue
        user = row['user']
        song = row['song']
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
        res_matrix[user_i, item_i] = play_count
        cnt += 1
        print('Download {}%'.format(cnt / float(data.shape[0]) * 100))
    print(count_skipp)

    with open('group_index_dictionary.tit', "wb") as f:
        pickle.dump(items, f)
    with open('users_index_dictionary.tit', "wb") as f:
        pickle.dump(users, f)
    with open('item_user_matrix.tit', "wb") as f:
        pickle.dump(res_matrix, f)


def create_related_matrix_by_cos(item_user_matrix):
    # нормализуем исходную матрицу
    normalized_matrix = normalize(item_user_matrix.tocsr()).tocsr()

    # вычисляем скалярное произведение
    cos_sim_matrix = normalized_matrix.T.dot(normalized_matrix)

    # обнуляем диагональ, чтобы исключить ее из рекомендаций
    diag = spdiags(-cos_sim_matrix.diagonal(), [0], *cos_sim_matrix.shape, format='csr')
    cos_sim_matrix = cos_sim_matrix + diag

    with open('cosine_sim_matrix_train.tit', "wb") as f:
        pickle.dump(cos_sim_matrix, f)

    return cos_sim_matrix


def grouping(cos_sim_matrix):
    from scipy.sparse import vstack

    cos_sim_matrix = cos_sim_matrix.tocsr()
    m = 30

    # построим top-k матрицу в один поток
    rows = []
    for row_id in np.unique(cos_sim_matrix.nonzero()[0]):
        row = cos_sim_matrix[row_id]  # исходная строка матрицы
        if row.nnz > m:
            work_row = row.tolil()
            # заменяем все top-k элементов на 0, результат отнимаем от row
            work_row[0, row.nonzero()[1][np.argsort(row.data)[-m:]]] = 0
            row = row - work_row.tocsr()
        rows.append(row)
    top_k_matrix = vstack(rows)
    # нормализуем матрицу-результат
    top_k_matrix = normalize(top_k_matrix)

    with open('top_k_matrix.tit', "wb") as f:
        pickle.dump(top_k_matrix, f)


def get_user_vector(data, item_item_matrix, user_new_id):
    user_vec = sparse.lil_matrix((item_item_matrix.shape[0], 1))
    rows, cols = data.nonzero()
    for row, col in zip(rows, cols):
        if row == user_new_id:
            user_vec[col, 0] = data[row, col]
    return user_vec


def rmse_evaluation(item_item_matrix, item_user_matrix, user_dictionary):
    from sklearn.metrics import mean_squared_error as mse
    from math import sqrt

    rmse_list = []
    counter = 0.0
    for user_id in list(user_dictionary.keys())[:200]:
        user_target_id = user_dictionary.get(user_id)
        user_vector = get_user_vector(item_user_matrix, item_item_matrix, user_target_id)
        x = item_item_matrix.T.dot(user_vector).tolil()
        real_list = []
        pred_list = []
        for i, j in zip(*user_vector.nonzero()):
            real_list.append(user_vector[i, j])
            pred_list.append(x[i, j])
            x[i, j] = 0
        rmse = sqrt(mse(real_list, pred_list))
        rmse_list.append(rmse)
        counter += 1
        print('Progress = {}%'.format(
            counter / len(list(user_dictionary.keys())[:100]) * 100))
    print(np.mean(rmse_list))


def prepare_the_data():
    transform_first_data()

    # обработка файла
    data_train = pd.read_csv('train_triplets_sub.csv', header=None)
    data_train.columns = ["user", "song", "play_count"]

    download_and_convert_data(data_train)

    with open('item_user_matrix.tit', "rb") as f:
        data = pickle.load(f)

    cosine_sim_matrix = create_related_matrix_by_cos(data)
    grouping(cosine_sim_matrix)


def main():
    while True:
        answer = input('Do you want change the data? [y/n]\n')
        if answer == 'y':
            prepare_the_data()
        elif answer == 'n':
            break
        else:
            continue

    with open('users_index_dictionary.tit', "rb") as f:
        users_dict = dict(pickle.load(f))

    with open('group_index_dictionary.tit', "rb") as f:
        items_dict = dict(pickle.load(f))

    print('User\'s dictionary of keys: {}'.format(
        list(i for i in users_dict.items() if i[1] < 100)))
    print('Song\'s dictionary of keys: {}'.format(
        list(i for i in items_dict.items() if i[1] < 100)))

    # вводим изначальный индекс пользователя
    # user_id = '403b3b867fc71dfdcc12652f30e88bdc7ccd9aa4'
    user_id = input(u'Enter user\'s id: \n')
    user_target_id = users_dict.get(user_id)

    # вводим изначальный индекс песни
    # song_id = 'SODLLYS12A8C13A96B'
    song_id = input(u'Enter songs\'s id: \n')
    song_target_id = items_dict.get(song_id)
    print('Await the results...')

    with open('item_user_matrix.tit', "rb") as f:
        item_user_matrix = pickle.load(f)

    with open('top_k_matrix.tit', "rb") as f:
        item_item_matrix = pickle.load(f)

    user_vector = get_user_vector(item_user_matrix, item_item_matrix, user_target_id)

    # перемножаем матрицу item-item и вектор рейтингов пользователя
    x = item_item_matrix.T.dot(user_vector).tolil()

    # зануляем ячейки, соответствующие песням, которые пользователь уже прослушивал
    for i, j in zip(*user_vector.nonzero()):
        x[i, j] = 0

    # столбец результата -> вектор
    x = x.T.tocsr()

    # сортируем песни в порядке убывания значений
    data_ids = np.argsort(x.data)[:]

    result = []
    for arg_id in data_ids:
        row_id, p = x.indices[arg_id], x.data[arg_id]
        song_key = [key for key, value in items_dict.items() if value == row_id][0]
        result.append({'song_id': song_key, 'weight': p})

    print('----------------------------------------')
    print('Amount of recommendations = {}'.format(len(result)))
    print('----------------------------------------')

    df = pd.DataFrame(list(reversed(sorted(result, key=lambda k: k['weight']))))
    print(df)
    df.to_csv('Recommendations_for_user_{user}.csv'.format(user=user_id))

    print('----------------------------------------')
    print('For user {user} weight of song {song} = {val}'.format(
        user=user_target_id,
        song=song_target_id,
        val=float(x[0, song_target_id])))

    # вычисляем rmse (~7)
    # rmse_evaluation(item_user_matrix, item_item_matrix, users_dict)

main()
