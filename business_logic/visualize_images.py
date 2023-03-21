from business_logic.preprocessing import preprocess, show
import numpy as np

data = np.load("data/data_train.npy")

test = data[:,-100:-1]
testp = preprocess(test)
for im in testp:
    show(im)