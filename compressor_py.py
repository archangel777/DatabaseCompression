
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from compressor_model import *

def plot_gauss(n0, n1, nStep, nRepeat, dataset, e):
    x = [i for i in range(n0, n1 + 1, nStep)]
    y = []
    for i in x:
        model = MyModel(n_gauss=i, err=e)
        model.load_data(dataset)
        mean = 0
        for j in range(nRepeat):
            model.train()
            model.parRun()
            mean += model.compression
        y.append(mean / nRepeat)

    print(y)

    plt.plot(x, y)
    plt.show()
    plt.savefig("testeTotalDB51.png")



# plot_gauss(1, 3, 1, 1, 'iris', e=0.01)
plot_gauss(1, 5, 1, 1, 'tpc-h-5k', e=0.01)

# In[ ]: