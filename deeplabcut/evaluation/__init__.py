"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

This script evaluates a trained model at a particular state on the data set
(images) and stores the results in a pandas dataframe.

Script called from Step1_EvaluateModelonDataset.py

"""

####################################################
# Dependencies
####################################################

from deeplabcut import myconfig
from deeplabcut import paths
from deeplabcut import utils

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
    print("Starting evaluation")

    shuffle = CONF.net.shuffles[shuffle_index]
    trainFraction = CONF.net.training_fraction[train_fraction_index]

    datafile = paths.get_train_docfile(trainFraction, shuffle)

    # loading meta data / i.e. training & test files & labels
    with open(datafile, 'rb') as f:
        data, trainIndices, testIndices, __ignore__ = pickle.load(f)

    Data = pd.read_hdf(paths.get_collected_data_file(CONF.labelling.scorer),
                       'df_with_missing')

    #######################################################################
    # Load and setup CNN part detector as well as its configuration
    #######################################################################

    cfg = load_config(paths.get_pose_cfg_test(trainFraction, shuffle))
    Snapshots = np.array(paths.get_train_snapshots(trainFraction, shuffle))

    increasing_indices = np.argsort([int(m.rsplit('-', 1)[1])
                                     for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]

    cfg['init_weights'] = Snapshots[snapshot_index]
    trainingsiterations = cfg['init_weights'].rsplit('-')[-1]
    DLCscorer = paths.get_scorer_name(cfg["net_type"],
                                      trainFraction,
                                      shuffle,
                                      trainingsiterations)
    print("Running ", DLCscorer,
          " with # of trainingiterations:", trainingsiterations)

    results_dir = paths.get_results_dir()
    utils.attempttomakefolder(results_dir)
    results_file = paths.get_scorer_file(cfg["net_type"],
                                         trainFraction,
                                         shuffle,
                                         trainingsiterations)

    try:
        Data = pd.read_hdf(results_file,
                           'df_with_missing')
        print("This net has already been evaluated!")
    except FileNotFoundError:
        # Specifying state of model (snapshot / training state)
        cfg['init_weights'] = Snapshots[snapshot_index]
        sess, inputs, outputs = predict.setup_pose_prediction(cfg)

        Numimages = len(Data.index)
        PredicteData = np.zeros((Numimages, 3 * len(cfg['all_joints_names'])))

        print("Analyzing data...")

        ##################################################
        # Compute predictions over images
        ##################################################

        for imageindex, imagename in tqdm(enumerate(Data.index)):
            image_path = paths.get_training_imagefile(imagename)

            image = io.imread(image_path, mode='RGB')
            image = skimage.color.gray2rgb(image)
            image_batch = data_to_input(image)
            # Compute prediction with the CNN
            outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
            scmap, locref = predict.extract_cnn_output(outputs_np, cfg)

            # Extract maximum scoring location from the heatmap, assume 1
            # person
            pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
            # NOTE: thereby     cfg_test['all_joints_names'] should be same
            # order as bodyparts!
            PredicteData[imageindex, :] = pose.flatten()

        index = pd.MultiIndex.from_product(
            [[DLCscorer], cfg['all_joints_names'], ['x', 'y', 'likelihood']],
            names=['scorer', 'bodyparts', 'coords'])

        # Saving results:
        DataMachine = pd.DataFrame(
            PredicteData, columns=index, index=Data.index.values
        )
        DataMachine.to_hdf(results_file,
                           'df_with_missing',
                           format='table',
                           mode='w')
        print("Done and results stored for snapshot: ",
              Snapshots[snapshot_index])
