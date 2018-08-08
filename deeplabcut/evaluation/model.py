"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

This script evaluates a trained model at a particular state on the data set
(images) and stores the results in a pandas dataframe.

You need tensorflow for evaluation. Run by:

CUDA_VISIBLE_DEVICES=0 python3 Step1_EvaluateModelonDataset.py

"""

####################################################
# Dependencies
####################################################

from deeplabcut import myconfig
from deeplabcut import paths
import numpy as np
import pandas as pd

# Deep-cut dependencies
from deeplabcut import evaluation

CONF = myconfig.CONF


def main():
    ####################################################
    # Loading data and evaluating network on data
    ####################################################

    for shuffleIndex, shuffle in enumerate(CONF.net.shuffles):
        for (trainFractionIndex,
             trainFraction) in enumerate(CONF.net.training_fraction):
            ###################################################################
            # Check which snapshots exist for given network (with training data
            # split).
            ###################################################################

            # Check which snap shots are available and sort them by #
            # iterations
            Snapshots = np.array(paths.get_train_snapshots(trainFraction,
                                                           shuffle))
            increasing_indices = np.argsort(
                [int(m.rsplit('-', 1)[1]) for m in Snapshots]
            )
            Snapshots = Snapshots[increasing_indices]

            snapshotindex = CONF.evaluation.snapshotindex
            if snapshotindex == "all":
                snapindices = range(len(Snapshots))
            else:
                snapshotindex = int(snapshotindex)
                if snapshotindex == -1:
                    snapindices = [-1]
                elif snapshotindex < len(Snapshots):
                    snapindices = [snapshotindex]
                else:
                    print("Invalid choice, only -1 (last), any integer "
                          "up to last, or all (as string)!")

            for snapIndex in snapindices:
                evaluation.evaluate_network(Snapshots[snapIndex], shuffleIndex,
                                            trainFractionIndex)
