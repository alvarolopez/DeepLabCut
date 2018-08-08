"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

"""
import errno
import os
import os.path
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from deeplabcut import myconfig
CONF = myconfig.CONF


def attempttomakefolder(foldername):
    try:
        os.makedirs(foldername)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(foldername):
            print("Folder already exists!", foldername)
        else:
            raise


def SaveData(PredicteData, metadata, dataname, pdindex, imagenames):
    DataMachine = pd.DataFrame(PredicteData, columns=pdindex, index=imagenames)
    DataMachine.to_hdf(dataname, 'df_with_missing', format='table', mode='w')
    with open(dataname.split('.')[0] + 'includingmetadata.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)


def get_immediate_subdirectories(a_dir):
    # https://stackoverflow.com/questions/800197/how-to-get-all-of-the-immediate-subdirectories-in-python
    return [
        name for name in os.listdir(a_dir)
        if os.path.isdir(os.path.join(a_dir, name))
    ]


def get_cmap(n, name=None):
    '''
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap
    name.
    '''
    if name is None:
        name = CONF.label.colormap
    return plt.cm.get_cmap(name, n)
