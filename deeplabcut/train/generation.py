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
import sys
import pandas as pd

import pickle
import shutil
import yaml

import scipy.io as sio

from deeplabcut import myconfig
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
    This script generates the training data information for DeepCut (which requires a mat file)
    based on the pandas dataframes that hold label information. The user can set the fraction of
    the traing set size (from all labeled image in the hd5 file) and can create multiple shuffles.
    """

    date = CONF.net.date

    ####################################################
    # Definitions (Folders, data source and labels)
    ####################################################

    task = CONF.data.task
    frame_folder = os.path.join(CONF.data.base_directory, "frames", task)
    label_folder = os.path.join(CONF.data.base_directory, "labels", task)
    tmp_folder = os.path.join(CONF.data.base_directory, "tmp", task)

    train_folder = os.path.join(CONF.data.base_directory, "train", task)
    utils.attempttomakefolder(train_folder)

    # Loading scorer's data:
    filename = 'CollectedData_' + CONF.label.scorer + '.h5'
    aux = os.path.join(label_folder, filename)
    Data = pd.read_hdf(aux, 'df_with_missing')[CONF.label.scorer]

    # Make that folder and put in the collecteddata (see below)
    bf = "UnaugmentedDataSet_" + task + date + "/"
    base_folder = os.path.join(train_folder, bf)
    print(label_folder)

#    # copy images and folder structure in the folder containing
#    # training data comparison
    shutil.copytree(frame_folder, os.path.join(base_folder, "frames"))
    shutil.copytree(label_folder, os.path.join(base_folder, "labels"))
    shutil.copytree(tmp_folder, os.path.join(base_folder, "labelled"))

    for shuffle in CONF.net.shuffles:
        for trainFraction in CONF.net.training_fraction:
            trainIndexes, testIndexes = SplitTrials(
                range(len(Data.index)), trainFraction)
            filename_matfile = task + "_" + CONF.label.scorer + str(int(
                100 * trainFraction)) + "shuffle" + str(shuffle)
            # Filename for pickle file:
            fn = os.path.join(base_folder, "Documentation_" + task + "_" + str(
                int(trainFraction * 100)) + "shuffle" + str(shuffle))

            ####################################################
            # Generating data structure with labeled information & frame metadata (for deep cut)
            ####################################################

            # Make matlab train file!
            data = []
            for jj in trainIndexes:
                H = {}
                # load image to get dimensions:
                filename = Data.index[jj]
                aux_path = os.path.relpath(filename, frame_folder)
                H['image'] = os.path.abspath(os.path.join(base_folder, "frames", aux_path))
                im = io.imread(H["image"])

                if np.ndim(im)>2:
                    H['size'] = np.array(
                        [np.shape(im)[2],
                         np.shape(im)[0],
                         np.shape(im)[1]])
                else:
                    # print "Grayscale!"
                    H['size'] = np.array([1, np.shape(im)[0], np.shape(im)[1]])

                indexjoints=0
                joints=np.zeros((len(CONF.dataframe.bodyparts),3))*np.nan
                for bpindex,bodypart in enumerate(CONF.dataframe.bodyparts):
                    if Data[bodypart]['x'][jj]<np.shape(im)[1] and Data[bodypart]['y'][jj]<np.shape(im)[0]: #are labels in image?
                            joints[indexjoints,0]=int(bpindex)
                            joints[indexjoints,1]=Data[bodypart]['x'][jj]
                            joints[indexjoints,2]=Data[bodypart]['y'][jj]
                            indexjoints+=1

                joints = joints[np.where(
                    np.prod(np.isfinite(joints),
                            1))[0], :]  # drop NaN, i.e. lines for missing body parts

                assert (np.prod(np.array(joints[:, 2]) < np.shape(im)[0])
                        )  # y coordinate within!
                assert (np.prod(np.array(joints[:, 1]) < np.shape(im)[1])
                        )  # x coordinate within!

                H['joints'] = np.array(joints, dtype=int)
                if np.size(joints)>0: #exclude images without labels
                        data.append(H)


            with open(fn + '.pickle', 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump([data, trainIndexes, testIndexes, trainFraction], f,
                            pickle.HIGHEST_PROTOCOL)

            ################################################################################
            # Convert to idosyncratic training file for deeper cut (*.mat)
            ################################################################################

            DTYPE = [('image', 'O'), ('size', 'O'), ('joints', 'O')]
            MatlabData = np.array(
                [(np.array([data[item]['image']], dtype='U'),
                  np.array([data[item]['size']]),
                  boxitintoacell(data[item]['joints']))
                 for item in range(len(data))],
                dtype=DTYPE)
            sio.savemat(os.path.join(base_folder, filename_matfile + '.mat'), {'dataset': MatlabData})

            ################################################################################
            # Creating file structure for training &
            # Test files as well as pose_yaml files (containing training and testing information)
            #################################################################################

            experimentname = task + date + '-trainset' + str(
                int(trainFraction * 100)) + 'shuffle' + str(shuffle)

            experiment_folder = os.path.join(train_folder, experimentname)
            utils.attempttomakefolder(experiment_folder)
            utils.attempttomakefolder(os.path.join(experiment_folder, 'train'))
            utils.attempttomakefolder(os.path.join(experiment_folder, 'test'))

            items2change = {
                "dataset": os.path.abspath(os.path.join(base_folder, filename_matfile + '.mat')),
                "num_joints": len(CONF.dataframe.bodyparts),
                "all_joints": [[i] for i in range(len(CONF.dataframe.bodyparts))],
                "all_joints_names": CONF.dataframe.bodyparts
            }

            trainingdata = MakeTrain_pose_yaml(
                items2change,
                os.path.join(experiment_folder, 'train', 'pose_cfg.yaml'),
                filename='pose_cfg.yaml')
            keys2save = [
                "dataset", "num_joints", "all_joints", "all_joints_names",
                "net_type", 'init_weights', 'global_scale', 'location_refinement',
                'locref_stdev'
            ]
            MakeTest_pose_yaml(trainingdata, keys2save,
                               os.path.join(experiment_folder, 'test', 'pose_cfg.yaml'))
