"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

This script evaluates a trained model at a particular state on the data set (images)
and stores the results in a pandas dataframe.

You need tensorflow for evaluation. Run by:

CUDA_VISIBLE_DEVICES=0 python3 Step1_EvaluateModelonDataset.py

"""

####################################################
# Dependencies
####################################################

import sys, os
import subprocess

#subfolder = os.getcwd().split('Evaluation-Tools')[0]
#sys.path.append(subfolder)
#
## add parent directory: (where nnet & config are!)
#sys.path.append(subfolder + "pose-tensorflow")
#sys.path.append(subfolder + "Generating_a_Training_Set")

from deeplabcut import myconfig
#from myconfig import Task, date, Shuffles, TrainingFraction, snapshotindex
import numpy as np
import pandas as pd
# Deep-cut dependencies
from deeplabcut.pose_tensorflow.config import load_config

from deeplabcut import evaluation

CONF = myconfig.CONF


def main():
    ####################################################
    # Loading data and evaluating network on data
    ####################################################

    base_folder = os.path.join(CONF.data.base_directory, "train", CONF.data.task)
    folder = os.path.join(base_folder, 'UnaugmentedDataSet_' + CONF.data.task + CONF.net.date)

    for shuffleIndex, shuffle in enumerate(CONF.net.shuffles):
        for trainFractionIndex, trainFraction in enumerate(CONF.net.training_fraction):
            ################################################################################
            # Check which snapshots exist for given network (with training data split).
            ################################################################################

            experimentname = CONF.data.task + CONF.net.date + '-trainset' + str(
                int(trainFraction * 100)) + 'shuffle' + str(shuffle)
            modelfolder = os.path.join(base_folder, experimentname)
            print (modelfolder)
            cfg = load_config(os.path.join(base_folder , experimentname , 'test' ,"pose_cfg.yaml"))
            # Check which snap shots are available and sort them by # iterations
            Snapshots = np.array([
                fn.split('.')[0]
                for fn in os.listdir(os.path.join(base_folder, experimentname , 'train'))
                if "index" in fn])

            increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
            Snapshots = Snapshots[increasing_indices]

            snapshotindex = CONF.evaluation.snapshotindex
            if snapshotindex == "all":
                snapindices = range(len(Snapshots))
            else:
                snapshotindex = int(snapshotindex)
                if snapshotindex == -1:
                    snapindices = [-1]
                elif snapshotindex<len(Snapshots):
                    snapindices=[snapshotindex]
                else:
                    print("Invalid choice, only -1 (last), any integer up to last, or all (as string)!")

            for snapIndex in snapindices:
                cfg['init_weights'] = os.path.join(modelfolder,'train',Snapshots[snapIndex])
                trainingsiterations = (cfg['init_weights'].split('/')[-1]).split('-')[-1]
                scorer = 'DeepCut' + "_" + str(cfg["net_type"]) + "_" + str(
                    int(trainFraction *
                        100)) + 'shuffle' + str(shuffle) + '_' + str(trainingsiterations) + "forTask_" + CONF.data.task

                print("Running ", scorer, " with # of trainingiterations:", trainingsiterations)

                results_dir = os.path.join(CONF.data.base_directory, "results")

                try:
                    Data = pd.read_hdf(os.path.join(results_dir, scorer + '.h5'),'df_with_missing')
                    print("This net has already been evaluated!")
                except FileNotFoundError:
                    evaluation.evaluate_network(snapIndex, shuffleIndex, trainFractionIndex)
