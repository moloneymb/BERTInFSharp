import numpy as np

import matplotlib.pyplot as plt


train = np.load("train.npy")

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

train = train[:1000]


plt.plot(train,label="loss")
plt.plot( np.hstack([np.zeros(45), running_mean(train,100)]),label="running mean 100")
plt.title("BERT Imdb train")
plt.legend(loc="upper right")
plt.show()

