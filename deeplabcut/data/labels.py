"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut

A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

A key point is to select diverse frames, which are typical for the behavior
you study that should be labeled. This helper script selects N frames uniformly
sampled from a particular video. Ideally you would also get data from different
session and different animals if those vary substantially. Note: this might not
yield diverse frames, if the behavior is sparsely distributed.

Individual images should not be too big (i.e. < 850 x 850 pixel). Although this
can be taken care of later as well, it is advisable to crop the frames, to
remove unnecessary parts of the video as much as possible.
"""

import imageio
imageio.plugins.ffmpeg.download()
import matplotlib
matplotlib.use('Agg')
from moviepy.editor import VideoFileClip
from skimage import io
from skimage.util import img_as_ubyte
import numpy as np
import os
import math
import sys


from deeplabcut import myconfig
from deeplabcut import utils

CONF = myconfig.CONF


###### Step 2

import numpy as np
import pandas as pd


def convert_labels_to_data_frame():
    """
    This function generates a data structure in pandas, that contains the
    (relative) physical address of the image as well as the labels. These data
    are extracted from the "labeling.csv" files that can be generated in a
    different file e.g.  ImageJ / Fiji
    """

    task = CONF.data.task
    frame_folder = os.path.join(CONF.data.base_directory, "frames", task)
    label_folder = os.path.join(CONF.data.base_directory, "labels", task)

    ###################################################
    # Code if all bodyparts (per folder are shared in one file)
    # This code below converts it into multiple csv files per body part & folder
    # Based on an idea by @sneakers-the-rat
    ###################################################

    # FIXME(aloga): check this
#    if CONF.dataframe.multibodypartsfile==True:
#        folders = [name for name in os.listdir(frame_folder) if os.path.isdir(os.path.join(basefolder, name))]
#        for folder in folders:
#            # load csv, iterate over nth value in a grouping by frame, save to bodyparts files
#            dframe = pd.read_csv(os.path.join(basefolder,folder,CONF.dataframe.multibodypartsfilename))
#            frame_grouped = dframe.groupby('Slice') #Note: the order of bodyparts list in myconfig and labels must be identical!
#            for i, bodypart in enumerate(bodyparts):
#                part_df = frame_grouped.nth(i)
#                part_fn =  part_fn = os.path.join(basefolder,folder,bodypart+'.csv')
#                part_df.to_csv(part_fn)

    ###################################################
    # Code if each bodypart has its own label file!
    ###################################################

    # Data frame to hold data of all data sets for different scorers,
    # bodyparts and images
    DataCombined = None
    for scorer in CONF.dataframe.scorers:
#        os.chdir(label_folder)
        # Make list of different video data sets / each one has its own folder
        folders = [
            videodatasets for videodatasets in os.listdir(frame_folder)
            if os.path.isdir(os.path.join(frame_folder, videodatasets))
        ]
        try:
            filename = 'CollectedData_' + scorer + '.h5'
            aux = os.path.join(label_folder, filename)
            print(aux)
            DataSingleUser = pd.read_hdf(aux, 'df_with_missing')
            numdistinctfolders = list(
                set([s.split('/')[0] for s in DataSingleUser.index
                     ]))  # NOTE: SLICING to eliminate multiindices!
            # print("found",len(folders),len(numdistinctfolders))
            if len(folders) > len(numdistinctfolders):
                DataSingleUsers = None
                print("Not all data converted!")
            else:
                print(scorer, "'s data already collected!")
                print(DataSingleUser.head())
        except FileNotFoundError:
            DataSingleUser = None

        if DataSingleUser is None:
            for folder in folders:
                frame_folder = os.path.join(frame_folder, folder)
                # sort image file names according to how they were stacked
                # files=np.sort([fn for fn in os.listdir(os.curdir)
                # if ("img" in fn and ".png" in fn and "_labelled" not in fn)])
                files = [
                    fn for fn in os.listdir(frame_folder)
                    if ("img" in fn and CONF.dataframe.imagetype in fn and "_labelled" not in fn)
                ]
                files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

                imageaddress = [os.path.join(frame_folder, f) for f in files]
                Data_onefolder = pd.DataFrame({'Image name': imageaddress})

                frame, Frame = None, None
                for bodypart in CONF.dataframe.bodyparts:
                    datafile = bodypart
                    datafile = os.path.join(label_folder, folder, datafile)
                    dframe = pd.read_csv(datafile + ".csv",sep=None,engine='python') #, sep='\t')
                    # NOTE(aloga): why is this being moved?
#                    try:
#                        dframe = pd.read_csv(datafile + ".xls",sep=None,engine='python') #, sep='\t')
#                    except FileNotFoundError:
#                        os.rename(datafile + ".csv", datafile + ".xls")
#                        dframe = pd.read_csv(datafile + ".xls",sep=None,engine='python') #, sep='\t')

                    # Note: If your csv file is not correctly loaded, then a common error is:
                    # "AttributeError: 'DataFrame' object has no attribute 'X'" or the corresponding error with Slice
                    # Try to make sure you specify the seperator of the csv file correctly. See https://github.com/AlexEMG/DeepLabCut/issues/10 for details.

                    if dframe.shape[0] != len(imageaddress):
                        new_index = pd.Index(
                            np.arange(len(files)) + 1, name='Slice')
                        dframe = dframe.set_index('Slice').reindex(new_index)
                        dframe = dframe.reset_index()

                    index = pd.MultiIndex.from_product(
                        [[scorer], [bodypart], ['x', 'y']],
                        names=['scorer', 'bodyparts', 'coords'])

                    Xrescaled = dframe.X.values.astype(float)
                    Yrescaled = dframe.Y.values.astype(float)

                    # get rid of values that are invisible >> thus user scored in left corner!
                    invisiblemarkersmask = (Xrescaled < CONF.dataframe.invisibleboundary) * (Yrescaled < CONF.dataframe.invisibleboundary)
                    Xrescaled[invisiblemarkersmask] = np.nan
                    Yrescaled[invisiblemarkersmask] = np.nan

                    if Frame is None:
                        # frame=pd.DataFrame(np.vstack([dframe.X,dframe.Y]).T, columns=index,index=imageaddress)
                        frame = pd.DataFrame(
                            np.vstack([Xrescaled, Yrescaled]).T,
                            columns=index,
                            index=imageaddress)
                        # print(frame.head())
                        Frame = frame
                    else:
                        frame = pd.DataFrame(
                            np.vstack([Xrescaled, Yrescaled]).T,
                            columns=index,
                            index=imageaddress)
                        Frame = pd.concat(
                            [Frame, frame],
                            axis=1)  # along bodyparts & scorer dimension

                # print("Done with folder ", folder)
                if DataSingleUser is None:
                    DataSingleUser = Frame
                else:
                    DataSingleUser = pd.concat(
                        [DataSingleUser, Frame], axis=0)  # along filenames!

            # Save data by this scorer
            filename = 'CollectedData_' + scorer + '.csv'
            aux = os.path.join(label_folder, filename)
            DataSingleUser.to_csv(aux)  # breaks multiindices HDF5 tables better!

            filename = 'CollectedData_' + scorer + '.h5'
            aux = os.path.join(label_folder, filename)
            DataSingleUser.to_hdf(aux,
                                  'df_with_missing',
                                  format='table',
                                  mode='w')

        print("Merging scorer's data.")


# Step 3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
def get_cmap(n, name=CONF.label.colormap):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def check_labels():
    """
    Generates training images with labels to check if annotation was done
    correctly/correctly loaded.
    """
    scale = CONF.label.scale
    msize = CONF.label.label_size
    alphavalue = CONF.label.alpha
    colormap = CONF.label.colormap

    ###################################################
    # Code if each bodypart has its own label file!
    ###################################################

    Labels = ['.', '+', '*']  # order of labels for different scorers

    #############################################
    # Make sure you update the train.yaml file!
    #############################################

    bodyparts = CONF.dataframe.bodyparts
    num_joints = len(bodyparts)
    all_joints = map(lambda j: [j], range(num_joints))
    all_joints_names = bodyparts


    Colorscheme = get_cmap(len(bodyparts))

    print(num_joints)
    print(all_joints)
    print(all_joints_names)


    task = CONF.data.task
    frame_folder = os.path.join(CONF.data.base_directory, "frames", task)
    label_folder = os.path.join(CONF.data.base_directory, "labels", task)

    numbodyparts = len(bodyparts)

    # Data frame to hold data of all data sets for different scorers, bodyparts and images
    DataCombined = None

#    os.chdir(basefolder)

    filename = 'CollectedData_' + CONF.label.scorer + '.h5'
    aux = os.path.join(label_folder, filename)
    DataCombined = pd.read_hdf(aux, 'df_with_missing')

    # Make list of different video data sets:
    folders = [
        videodatasets for videodatasets in os.listdir(frame_folder)
        if os.path.isdir(os.path.join(frame_folder, videodatasets))
    ]

    print(folders)
    # videos=np.sort([fn for fn in os.listdir(os.curdir) if ("avi" in fn)])

    for folder in folders:
        tmp_folder = os.path.join(CONF.data.base_directory, "tmp", task, folder)
        utils.attempttomakefolder(tmp_folder)
        frame_folder = os.path.join(frame_folder, folder)
        # sort image file names according to how they were stacked (when labeled in Fiji)
        files = [
            fn for fn in os.listdir(frame_folder)
            if ("img" in fn and CONF.dataframe.imagetype in fn and "_labelled" not in fn)
        ]

        comparisonbodyparts = bodyparts #list(set(DataCombined.columns.get_level_values(1)))

        for index, imagename in enumerate(files):
            image = io.imread(os.path.join(frame_folder, imagename))
            plt.axis('off')

            if np.ndim(image)==2:
                h, w = np.shape(image)
            else:
                h, w, nc = np.shape(image)

            plt.figure(
                frameon=False, figsize=(w * 1. / 100 * scale, h * 1. / 100 * scale))
            plt.subplots_adjust(
                left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

            # This is important when using data combined / which runs consecutively!
            aux = os.path.join(os.path.join(frame_folder, imagename))
            imindex = np.where(
                np.array(DataCombined.index.values) == aux)[0]

            plt.imshow(image, 'bone')
            for cc, scorer in enumerate(CONF.dataframe.scorers):
                if index==0:
                    print("Creating images with labels by ", scorer)
                for c, bp in enumerate(comparisonbodyparts):
                    plt.plot(
                        DataCombined[scorer][bp]['x'].values[imindex],
                        DataCombined[scorer][bp]['y'].values[imindex],
                        Labels[cc],
                        color=Colorscheme(c),
                        alpha=alphavalue,
                        ms=msize)

            plt.xlim(0, w)
            plt.ylim(0, h)
            plt.axis('off')
            plt.subplots_adjust(
                left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.gca().invert_yaxis()
            plt.savefig(os.path.join(tmp_folder, imagename))
            plt.close("all")
