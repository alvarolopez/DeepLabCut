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

import os

from moviepy.editor import VideoFileClip
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io

from deeplabcut import myconfig
from deeplabcut import paths
from deeplabcut import utils

matplotlib.use('Agg')
CONF = myconfig.CONF


def convert_labels_to_data_frame():
    """
    This function generates a data structure in pandas, that contains the
    (relative) physical address of the image as well as the labels. These data
    are extracted from the "labeling.csv" files that can be generated in a
    different file e.g.  ImageJ / Fiji
    """

    label_folder = paths.get_label_dir()

    ###################################################
    # Code if all bodyparts (per folder are shared in one file)
    # This code below converts it into multiple csv files per body part &
    # folder
    # Based on an idea by @sneakers-the-rat
    ###################################################

    # FIXME(aloga): check this
#    if CONF.labelling.multibodypartsfile==True:
#        folders = [name for name in os.listdir(frame_folder)
#                   if os.path.isdir(os.path.join(basefolder, name))]
#        for folder in folders:
#            # load csv, iterate over nth value in a grouping by frame, save to
#            bodyparts files
#            dframe = pd.read_csv(os.path.join(
#                   basefolder,afolder,CONF.labelling.multibodypartsfilename))
# Note: the order of bodyparts list in myconfig and labels must be identical!
#            frame_grouped = dframe.groupby('Slice')
#            for i, bodypart in enumerate(bodyparts):
#                part_df = frame_grouped.nth(i)
#                part_fn =  part_fn = os.path.join(basefolder,
#                                                  folder, bodypart+'.csv')
#                part_df.to_csv(part_fn)

    ###################################################
    # Code if each bodypart has its own label file!
    ###################################################

    scorer = CONF.labelling.scorer
    # Make list of different video data sets / each one has its own folder
    folders = paths.get_video_datasets()
    try:
        filename = paths.get_collected_data_file(scorer)
        DataSingleUser = pd.read_hdf(filename, 'df_with_missing')
        # NOTE: SLICING to eliminate multiindices!
        numdistinctfolders = list(
            set([s.split('/')[0] for s in DataSingleUser.index
                    ]))
        # print("found",len(folders),len(numdistinctfolders))
        if len(folders) > len(numdistinctfolders):
            DataSingleUser = None
            print("Not all data converted!")
        else:
            print(scorer, "'s data already collected!")
            print(DataSingleUser.head())
    except FileNotFoundError:
        DataSingleUser = None

    if DataSingleUser is None:
        for folder in folders:
            files = paths.get_video_dataset_frames(folder)

            frame, Frame = None, None
            for bodypart in CONF.labelling.bodyparts:
                datafile = bodypart
                datafile = os.path.join(label_folder, folder, datafile)
                dframe = pd.read_csv(datafile + ".csv",
                                        # FIXME(aloga) add csv separator
                                        sep=None,
                                        engine='python')

                if dframe.shape[0] != len(files):
                    new_index = pd.Index(
                        np.arange(len(files)) + 1, name='Slice')
                    dframe = dframe.set_index('Slice').reindex(new_index)
                    dframe = dframe.reset_index()

                index = pd.MultiIndex.from_product(
                    [[scorer], [bodypart], ['x', 'y']],
                    names=['scorer', 'bodyparts', 'coords'])

                Xrescaled = dframe.X.values.astype(float)
                Yrescaled = dframe.Y.values.astype(float)

                # get rid of values that are invisible >> thus user scored
                # in left corner!
                invisiblemarkersmask = (
                    Xrescaled < CONF.labelling.invisibleboundary) * (
                        Yrescaled < CONF.labelling.invisibleboundary)
                Xrescaled[invisiblemarkersmask] = np.nan
                Yrescaled[invisiblemarkersmask] = np.nan

                if Frame is None:
                    # frame=pd.DataFrame(np.vstack([dframe.X,dframe.Y]).T,
                    # columns=index,index=files)
                    frame = pd.DataFrame(
                        np.vstack([Xrescaled, Yrescaled]).T,
                        columns=index,
                        index=files)
                    # print(frame.head())
                    Frame = frame
                else:
                    frame = pd.DataFrame(
                        np.vstack([Xrescaled, Yrescaled]).T,
                        columns=index,
                        index=files)
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
        filename = paths.get_collected_data_file(scorer, filetype='.csv')
        # breaks multiindices HDF5 tables better!
        DataSingleUser.to_csv(filename)

        filename = paths.get_collected_data_file(scorer)
        DataSingleUser.to_hdf(filename,
                                'df_with_missing',
                                format='table',
                                mode='w')

        print("Merging scorer's data.")


# Step 3


def check_labels():
    """
    Generates training images with labels to check if annotation was done
    correctly/correctly loaded.
    """
    scale = CONF.label.scale
    msize = CONF.label.label_size
    alphavalue = CONF.label.alpha
#    colormap = CONF.label.colormap

    bodyparts = CONF.labelling.bodyparts
    num_joints = len(bodyparts)
    all_joints = map(lambda j: [j], range(num_joints))
    all_joints_names = bodyparts

    Colorscheme = utils.get_cmap(len(bodyparts))

#    print(num_joints)
#    print(all_joints)
    print("Data was collected for", all_joints_names)

    # Data frame to hold data of all data sets for different scorers, bodyparts
    # and images
    DataCombined = None

    scorer = CONF.labelling.scorer
    filename = paths.get_collected_data_file(scorer)
    DataCombined = pd.read_hdf(filename, 'df_with_missing')

    # Make list of different video data sets in frame folder:
    folders = paths.get_video_datasets()

    # videos=np.sort([fn for fn in os.listdir(os.curdir) if ("avi" in fn)])

    # Create evaluation images for each of the folders
    for folder in folders:
        # Store data in a tmp directory with the same folder name
        tmp_folder = paths.get_tmp_dir(folder)
        utils.attempttomakefolder(tmp_folder)

        files = paths.get_video_dataset_frames(folder)

        comparisonbodyparts = bodyparts
        # list(set(DataCombined.columns.get_level_values(1)))

        # Read images in the folder
        for index, image_path in enumerate(files):
            image = io.imread(image_path)
            plt.axis('off')

            if np.ndim(image) == 2:
                h, w = np.shape(image)
            else:
                h, w, nc = np.shape(image)

            plt.figure(
                frameon=False,
                figsize=(w * 1. / 100 * scale, h * 1. / 100 * scale)
            )
            plt.subplots_adjust(
                left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

            # This is important when using data combined / which runs
            # consecutively!
            imindex = np.where(
                np.array(DataCombined.index.values) == image_path)[0]

            plt.imshow(image, 'bone')
            if index == 0:
                print("Creating images with labels by", scorer,
                      "from", filename)
            for c, bp in enumerate(comparisonbodyparts):
                plt.plot(
                    DataCombined[scorer][bp]['x'].values[imindex],
                    DataCombined[scorer][bp]['y'].values[imindex],
                    CONF.label.label,
                    color=Colorscheme(c),
                    alpha=alphavalue,
                    ms=msize)

            plt.xlim(0, w)
            plt.ylim(0, h)
            plt.axis('off')
            plt.subplots_adjust(
                left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.gca().invert_yaxis()

            image_name = os.path.basename(image_path)
            plt.savefig(os.path.join(tmp_folder, image_name))
            plt.close("all")
