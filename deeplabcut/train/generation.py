"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu
"""

####################################################
# Loading dependencies
####################################################

import numpy as np
import scipy.io as sio
from skimage import io
import os
import yaml
import pickle
import shutil
import pandas as pd

from deeplabcut import myconfig
from deeplabcut import paths
from deeplabcut import utils

CONF = myconfig.CONF


def SplitTrials(trialindex, trainFraction=0.8):
    ''' Split a trial index into train and test sets'''
    trainsize = int(len(trialindex) * trainFraction)
    shuffle = np.random.permutation(trialindex)
    testIndexes = shuffle[trainsize:]
    trainIndexes = shuffle[:trainsize]

    return (trainIndexes, testIndexes)


def boxitintoacell(joints):
    ''' Auxiliary function for creating matfile.'''
    outer = np.array([[None]], dtype=object)
    outer[0, 0] = np.array(joints, dtype='int64')
    return outer


def MakeTrain_pose_yaml(itemstochange, saveasfile, filename='pose_cfg.yaml'):
    # FIXME: we need to change this!!
    filename = os.path.join(os.path.dirname(__file__), filename)
    raw = open(filename).read()
    docs = []
    for raw_doc in raw.split('\n---'):
        try:
            docs.append(yaml.load(raw_doc))
        except SyntaxError:
            docs.append(raw_doc)

    for key in itemstochange.keys():
        docs[0][key] = itemstochange[key]

    with open(saveasfile, "w") as f:
        yaml.dump(docs[0], f)
    return docs[0]


def MakeTest_pose_yaml(dictionary, keys2save, saveasfile):
    dict_test = {}
    for key in keys2save:
        dict_test[key] = dictionary[key]

    dict_test['scoremap_dir'] = 'test'
    with open(saveasfile, "w") as f:
        yaml.dump(dict_test, f)


def generate_training_file_from_labelled_data():
    """
    This script generates the training data information for DeepCut (which
    requires a mat file) based on the pandas dataframes that hold label
    information. The user can set the fraction of the traing set size (from all
    labeled image in the hd5 file) and can create multiple shuffles.
    """

    date = CONF.net.date

    ####################################################
    # Definitions (Folders, data source and labels)
    ####################################################

    task = CONF.data.task
    frame_folder = paths.frame_dir
    label_folder = paths.label_dir
    tmp_folder = paths.tmp_dir
    train_folder = paths.train_dir
    utils.attempttomakefolder(train_folder)

    # Loading scorer's data:
    Data = pd.read_hdf(paths.get_collected_data_file(CONF.label.scorer),
                       'df_with_missing')[CONF.label.scorer]

    base_folder = paths.get_train_dataset_dir()
    utils.attempttomakefolder(base_folder)

#    # copy images and folder structure in the folder containing
#    # training data comparison
    shutil.copytree(frame_folder, os.path.join(base_folder, "frames"))
    shutil.copytree(label_folder, os.path.join(base_folder, "labels"))
    shutil.copytree(tmp_folder, os.path.join(base_folder, "labelled"))

    for shuffle in CONF.net.shuffles:
        for trainFraction in CONF.net.training_fraction:
            trainIndexes, testIndexes = SplitTrials(
                range(len(Data.index)), trainFraction)
            filename_matfile = paths.get_train_matfile(trainFraction, shuffle)
            # Filename for pickle file:
            docfile = paths.get_train_docfile(trainFraction, shuffle)

            ####################################################
            # Generating data structure with labeled information & frame
            # metadata (for deep cut)
            ####################################################

            # Make matlab train file!
            data = []
            for jj in trainIndexes:
                H = {}
                # load image to get dimensions:
                orig_filename = Data.index[jj]
                H['image'] = paths.get_training_imagefile(orig_filename)
                im = io.imread(H["image"])

                if np.ndim(im) > 2:
                    H['size'] = np.array(
                        [np.shape(im)[2],
                         np.shape(im)[0],
                         np.shape(im)[1]])
                else:
                    # print "Grayscale!"
                    H['size'] = np.array([1, np.shape(im)[0], np.shape(im)[1]])

                indexjoints = 0
                joints = np.zeros((len(CONF.dataframe.bodyparts), 3)) * np.nan
                for bpindex, bodypart in enumerate(CONF.dataframe.bodyparts):
                    # are labels in image?
                    if (Data[bodypart]['x'][jj] < np.shape(im)[1] and
                            Data[bodypart]['y'][jj] < np.shape(im)[0]):
                        joints[indexjoints, 0] = int(bpindex)
                        joints[indexjoints, 1] = Data[bodypart]['x'][jj]
                        joints[indexjoints, 2] = Data[bodypart]['y'][jj]
                        indexjoints += 1

                # drop NaN, i.e. lines for missing body parts
                joints = joints[np.where(
                    np.prod(np.isfinite(joints),
                            1))[0], :]

                # y coordinate within!
                assert (np.prod(np.array(joints[:, 2]) < np.shape(im)[0]))
                # x coordinate within!
                assert (np.prod(np.array(joints[:, 1]) < np.shape(im)[1]))

                H['joints'] = np.array(joints, dtype=int)
                # exclude images without labels
                if np.size(joints) > 0:
                    data.append(H)

            with open(docfile, 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol
                # available.
                pickle.dump([data, trainIndexes, testIndexes, trainFraction],
                            f,
                            pickle.HIGHEST_PROTOCOL)

            ###################################################################
            # Convert to idosyncratic training file for deeper cut (*.mat)
            ###################################################################

            DTYPE = [('image', 'O'), ('size', 'O'), ('joints', 'O')]
            MatlabData = np.array(
                [(np.array([data[item]['image']], dtype='U'),
                  np.array([data[item]['size']]),
                  boxitintoacell(data[item]['joints']))
                 for item in range(len(data))],
                dtype=DTYPE)
            sio.savemat(filename_matfile, {'dataset': MatlabData})

            ##################################################################
            # Creating file structure for training &
            # Test files as well as pose_yaml files (containing training and
            # testing information)
            ##################################################################

            experiment_folder = paths.get_experiment_name(trainFraction, shuffle)
            utils.attempttomakefolder(experiment_folder)
            utils.attempttomakefolder(os.path.join(experiment_folder, 'train'))
            utils.attempttomakefolder(os.path.join(experiment_folder, 'test'))

            items2change = {
                "dataset": filename_matfile,
                "num_joints": len(CONF.dataframe.bodyparts),
                "init_weights": paths.get_pre_trained_file(),
                "all_joints": [
                    [i] for i in range(len(CONF.dataframe.bodyparts))
                ],
                "all_joints_names": CONF.dataframe.bodyparts
            }

            trainingdata = MakeTrain_pose_yaml(
                items2change,
                paths.get_pose_cfg_train(trainFraction, shuffle),
                filename="pose_cfg.yaml"
            )
            keys2save = [
                "dataset", "num_joints", "all_joints", "all_joints_names",
                "net_type", 'init_weights', 'global_scale',
                'location_refinement', 'locref_stdev'
            ]
            MakeTest_pose_yaml(
                trainingdata,
                keys2save,
                paths.get_pose_cfg_test(trainFraction, shuffle)
            )
