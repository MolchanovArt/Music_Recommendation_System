# only python 2.X
import os
import recsys.algorithm
from recsys.algorithm.factorize import SVD
from recsys.datamodel.data import Data
from recsys.evaluation.prediction import RMSE, MAE

recsys.algorithm.VERBOSE = True
file_path = os.path.expanduser('res_data_with_keys.csv')
# PERCENT_TRAIN = [65, 70, 75, 80]
PERCENT_TRAIN = [75]

data = Data()
data.load(path=file_path, sep=',', format={'col': 0, 'row': 1, 'value': 2})


for percent in PERCENT_TRAIN:
    # Train & Test data
    train, test = data.split_train_test(percent=percent)

    # Create simple SVD
    K = 100
    svd = SVD()
    svd.set_data(train)
    svd.compute(k=K,
                mean_center=True,
                post_normalize=True,
                # savefile='/home/mart/models_{percent}'.format(percent=percent)
                savefile='./models_{percent}'.format(percent=percent)
                )

    # Evaluation
    rmse = RMSE()
    mae = MAE()
    for rating, item_id, user_id in test.get():
        try:
            pred_rating = svd.predict(item_id, user_id)
            rmse.add(rating, pred_rating)
            mae.add(rating, pred_rating)
        except KeyError:
            continue

    print 'Percent = {percent}'.format(percent=percent)
    print 'RMSE={rmse}'.format(rmse=rmse.compute())
    print 'MAE={mae}'.format(mae=mae.compute())
