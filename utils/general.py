import os, matplotlib, yaml
from matplotlib import pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass

def visualize(batch64, savepath=None):
    if matplotlib.rcParams['backend'].lower() != 'agg': matplotlib.use('agg')
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(np.transpose(make_grid(batch64, padding=2, normalize=True).cpu(),(1,2,0)))
    if savepath is None: plt.show()
    else: plt.savefig(savepath)
    plt.close()

def load_yaml(path):
    file = open(path, 'r')
    options = yaml.load(file, Loader=yaml.FullLoader)
    file.close()
    return options