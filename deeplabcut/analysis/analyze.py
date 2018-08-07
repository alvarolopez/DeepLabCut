"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut

A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

This script analyzes videos based on a trained network (as specified in
myconfig_analysis.py)

You need tensorflow for evaluation. Run by:

CUDA_VISIBLE_DEVICES=0 python3 AnalyzeVideos.py

"""

####################################################
# Dependencies
####################################################

import pickle

# Deep-cut dependencies
from deeplabcut.pose_tensorflow.config import load_config
from deeplabcut.pose_tensorflow.nnet import predict
from deeplabcut.pose_tensorflow.dataset.pose_dataset import data_to_input

from skimage.util import img_as_ubyte
from moviepy.editor import VideoFileClip
import skimage
import skimage.color
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

from deeplabcut import myconfig
from deeplabcut import paths
from deeplabcut import utils

CONF = myconfig.CONF


def getpose(image, cfg, sess, inputs, outputs, outall=False):
    ''' Adapted from DeeperCut, see pose-tensorflow folder'''
    image_batch = data_to_input(skimage.color.gray2rgb(image))
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref = predict.extract_cnn_output(outputs_np, cfg)
    pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
    if outall:
        return scmap, locref, pose
    else:
        return pose


def main(videofolder=None):
    myconfig.parse_args()

    ####################################################
    # Loading data, and defining model folder
    ####################################################

    cfg = load_config(paths.get_pose_cfg_test(CONF.analysis.trainings_fraction,
                                              CONF.analysis.shuffle_index))

    ##################################################
    # Load and setup CNN part detector
    ##################################################

    # Check which snap shots are available and sort them by # iterations
    Snapshots = np.array(
        paths.get_train_snapshots(CONF.analysis.trainings_fraction,
                                  CONF.analysis.shuffle_index)
    )
    increasing_indices = np.argsort(
        [int(m.rsplit('-', 1)[1]) for m in Snapshots]
    )
    Snapshots = Snapshots[increasing_indices]

    ##################################################
    # Compute predictions over images
    ##################################################

    # Check if data already was generated:
    cfg['init_weights'] = Snapshots[CONF.analysis.snapshot_index]

    # Name for scorer:
    trainingsiterations = cfg['init_weights'].rsplit('-', 1)[-1]

    # Name for scorer based on passed on parameters from myconfig_analysis.
    # Make sure they refer to the network of interest.
    scorer = paths.get_scorer_name(CONF.net.resnet,
                                   CONF.analysis.trainings_fraction,
                                   CONF.analysis.shuffle_index,
                                   trainingsiterations)

    sess, inputs, outputs = predict.setup_pose_prediction(cfg)
    pdindex = pd.MultiIndex.from_product(
        [[scorer], cfg['all_joints_names'], ['x', 'y', 'likelihood']],
        names=['scorer', 'bodyparts', 'coords'])

    frame_buffer = 10

    if videofolder is None:
        videofolder = paths.get_video_dir()

    videos = paths.get_videos(videofolder)
    print("Starting ", videos)
    for video in videos:
        outdir = paths.get_video_outdir(video)
        utils.attempttomakefolder(outdir)
        dataname = paths.get_video_dataname(video, scorer)
        try:
            # Attempt to load data...
            pd.read_hdf(dataname)
            print("Video already analyzed!", dataname)
        except FileNotFoundError:
            print("Loading ", video)
            clip = VideoFileClip(video)
            ny, nx = clip.size  # dimensions of frame (height, width)
            fps = clip.fps
            # this is slow (but accurate)
            # nframes = np.sum(1 for j in clip.iter_frames())
            nframes_approx = int(np.ceil(clip.duration * clip.fps) +
                                 frame_buffer)
            # this will overestimage number of frames (see
            # https://github.com/AlexEMG/DeepLabCut/issues/9) This is
            # especially a problem for high frame rates and long durations due
            # to rounding errors (as Rich Warren found). Later we crop the
            # result (line 187)

            if CONF.data.cropping:
                clip = clip.crop(
                    y1=CONF.data.y1,
                    y2=CONF.data.y2,
                    x1=CONF.data.x1,
                    x2=CONF.data.x2)

            print("Duration of video [s]: ", clip.duration,
                  ", recorded with ", fps, "fps!")
            print("Overall # of frames: ", nframes_approx,
                  "with cropped frame dimensions: ", clip.size)

            start = time.time()
            PredicteData = np.zeros((nframes_approx,
                                     3 * len(cfg['all_joints_names'])))
            clip.reader.initialize()
            print("Starting to extract posture")
            for index in tqdm(range(nframes_approx)):
                # image = img_as_ubyte(clip.get_frame(index * 1. / fps))
                image = img_as_ubyte(clip.reader.read_frame())
                # Thanks to Rick Warren for the  following snipplet:
                # if close to end of video, start checking whether two adjacent
                # frames are identical this should only happen when moviepy has
                # reached the final frame if two adjacent frames are identical,
                # terminate the loop
                if index == int(nframes_approx - frame_buffer * 2):
                    last_image = image
                elif index > int(nframes_approx - frame_buffer * 2):
                    if (image == last_image).all():
                        nframes = index
                        print("Detected frames: ", nframes)
                        break
                    else:
                        last_image = image
                pose = getpose(image, cfg, sess, inputs, outputs)
                # NOTE: thereby cfg['all_joints_names'] should be same order as
                # bodyparts!
                PredicteData[index, :] = pose.flatten()

            stop = time.time()

            dictionary = {
                "start": start,
                "stop": stop,
                "run_duration": stop - start,
                "Scorer": scorer,
                "config file": cfg,
                "fps": fps,
                "frame_dimensions": (ny, nx),
                "nframes": nframes
            }
            metadata = {'data': dictionary}

            print("Saving results...")
            # slice pose data to have same # as # of frames.
            DataMachine = pd.DataFrame(PredicteData[:nframes, :],
                                       columns=pdindex,
                                       index=range(nframes))
            DataMachine.to_hdf(dataname,
                               'df_with_missing',
                               format='table',
                               mode='w')

            if CONF.analysis.store_as_csv:
                DataMachine.to_csv(video.split('.')[0] + scorer + '.csv')

            with open(dataname.split('.')[0] + 'includingmetadata.pickle',
                      'wb') as f:
                pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
