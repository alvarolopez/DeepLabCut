"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

This script evaluates a trained model at a particular state on the data set (images)
and stores the results in a pandas dataframe.

Script called from Step1_EvaluateModelonDataset.py

"""

####################################################
# Dependencies
####################################################

import sys
import os
#subfolder = os.getcwd().split('Evaluation-Tools')[0]
#sys.path.append(subfolder)
#
## add parent directory: (where nnet & config are!)
#sys.path.append(subfolder + "pose-tensorflow")
#sys.path.append(subfolder + "Generating_a_Training_Set")

#from myconfig import Task, date, Shuffles, scorer, TrainingFraction
from deeplabcut import myconfig
from deeplabcut.train import auxiliaryfunctions

# Deep-cut dependencies
from deeplabcut.pose_tensorflow.config import load_config
from deeplabcut.pose_tensorflow.nnet import predict
from deeplabcut.pose_tensorflow.dataset.pose_dataset import data_to_input

# Dependencies for anaysis
import pickle
import skimage
import numpy as np
import pandas as pd
from skimage import io
import skimage.color
from tqdm import tqdm

CONF = myconfig.CONF


def evaluate_network(snapshot_index, shuffle_index, train_fraction_index):
    print("Starting evaluation") #, sys.argv)
#    snapshot_index=int(sys.argv[1])
#    shuffle_index=int(sys.argv[2])
#    train_fraction_index=int(sys.argv[3])

    shuffle = CONF.net.shuffles[shuffle_index]
    trainFraction = CONF.net.training_fraction[train_fraction_index]

#    basefolder = os.path.join('..','pose-tensorflow','models')
#    folder = os.path.join('UnaugmentedDataSet_' + CONF.data.task + CONF.net.date)
    base_folder = os.path.join(CONF.data.base_directory, "train", CONF.data.task)
    folder = os.path.join(base_folder, 'UnaugmentedDataSet_' + CONF.data.task + CONF.net.date)

    datafile = ('Documentation_' + CONF.data.task + '_' +
                str(int(CONF.net.training_fraction[train_fraction_index] * 100)) + 'shuffle' +
                str(int(CONF.net.shuffles[shuffle_index])) + '.pickle')

    print(folder)
    # loading meta data / i.e. training & test files & labels
    with open(os.path.join(folder ,datafile), 'rb') as f:
        data, trainIndices, testIndices, __ignore__ = pickle.load(f)

    Data = pd.read_hdf(os.path.join(folder, "labels", 'CollectedData_' + CONF.label.scorer + '.h5'),'df_with_missing')

    #######################################################################
    # Load and setup CNN part detector as well as its configuration
    #######################################################################

    experimentname = CONF.data.task + CONF.net.date + '-trainset' + str(int(trainFraction * 100)) + 'shuffle' + str(shuffle)
    cfg = load_config(os.path.join(base_folder , experimentname , 'test' ,"pose_cfg.yaml"))
    modelfolder = os.path.join(base_folder, experimentname)

    Snapshots = np.array([fn.split('.')[0]
        for fn in os.listdir(os.path.join(base_folder, experimentname , 'train'))
        if "index" in fn
    ])
    increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]

    cfg['init_weights'] = os.path.join(modelfolder, 'train', Snapshots[snapshot_index])
    trainingsiterations = (
        cfg['init_weights'].split('/')[-1]).split('-')[-1]
    DLCscorer = 'DeepCut' + "_" + str(cfg["net_type"]) + "_" + str(
        int(trainFraction *
            100)) + 'shuffle' + str(shuffle) + '_' + str(trainingsiterations) + "forTask_" + CONF.data.task

    print("Running ", DLCscorer, " with # of trainingiterations:", trainingsiterations)

    results_dir = os.path.join(CONF.data.base_directory, "results")

    try:
        Data = pd.read_hdf(os.path.join(results_dir, DLCscorer + '.h5'),'df_with_missing')
        print("This net has already been evaluated!")
    except FileNotFoundError:
        # Specifying state of model (snapshot / training state)
        cfg['init_weights'] = os.path.join(modelfolder,'train',Snapshots[snapshot_index])
        sess, inputs, outputs = predict.setup_pose_prediction(cfg)

        Numimages = len(Data.index)
        PredicteData = np.zeros((Numimages,3 * len(cfg['all_joints_names'])))
        Testset = np.zeros(Numimages)

        print("Analyzing data...")

        ##################################################
        # Compute predictions over images
        ##################################################

        frame_folder = os.path.join(CONF.data.base_directory, "frames", CONF.data.task)

        for imageindex, imagename in tqdm(enumerate(Data.index)):
            aux_path = os.path.relpath(imagename, frame_folder)
            aux_path = os.path.abspath(os.path.join(folder, "frames", aux_path))

            image = io.imread(aux_path, mode='RGB')
            image = skimage.color.gray2rgb(image)
            image_batch = data_to_input(image)
            # Compute prediction with the CNN
            outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
            scmap, locref = predict.extract_cnn_output(outputs_np, cfg)

            # Extract maximum scoring location from the heatmap, assume 1 person
            pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
            PredicteData[imageindex, :] = pose.flatten(
            )  # NOTE: thereby     cfg_test['all_joints_names'] should be same order as bodyparts!

        index = pd.MultiIndex.from_product(
            [[DLCscorer], cfg['all_joints_names'], ['x', 'y', 'likelihood']],
            names=['scorer', 'bodyparts', 'coords'])

        # Saving results:
        auxiliaryfunctions.attempttomakefolder(results_dir)

        DataMachine = pd.DataFrame(
            PredicteData, columns=index, index=Data.index.values)
        DataMachine.to_hdf(os.path.join(results_dir, DLCscorer + '.h5'),'df_with_missing',format='table',mode='w')
        print("Done and results stored for snapshot: ", Snapshots[snapshot_index])
