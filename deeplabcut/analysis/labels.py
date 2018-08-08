"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut

A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

This script labels the bodyparts in videos as analzyed by "AnalyzeVideos.py".
This code is relatively slow as it stores all individual frames. Use
MakingLabeledVideo_fast.py instead for faster (and slightly different) version
(frames are not stored).

python3 MakingLabeledVideo.py

Note: run python3 AnalyzeVideos.py first.
"""

####################################################
# Dependencies
####################################################
import os.path
import matplotlib
import matplotlib.pyplot as plt
import imageio
from skimage.util import img_as_ubyte
from moviepy.editor import VideoFileClip
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import shutil
import tempfile

from deeplabcut import myconfig
from deeplabcut import paths

CONF = myconfig.CONF

matplotlib.use('Agg')
imageio.plugins.ffmpeg.download()


def get_cmap(n, name=CONF.label.colormap):
    '''
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap
    name.
    '''
    return plt.cm.get_cmap(name, n)


def CreateVideo(clip, Dataframe, outdir, tmpfolder, vname):
    '''Creating individual frames with labeled body parts and making a video'''
    scorer = np.unique(Dataframe.columns.get_level_values(0))[0]
    bodyparts2plot = list(np.unique(Dataframe.columns.get_level_values(1)))
    colors = get_cmap(len(bodyparts2plot))

    ny, nx = clip.size  # dimensions of frame (height, width)
    fps = clip.fps
    nframes = len(Dataframe.index)
    if CONF.video.cropping:
        # one might want to adjust
        clip = clip.crop(y1=CONF.video.y1,
                         y2=CONF.video.y2,
                         x1=CONF.video.x1,
                         x2=CONF.video.x2)
    clip.reader.initialize()
    print("Duration of video [s]: ", clip.duration, ", recorded with ", fps,
          "fps!")
    print("Overall # of frames: ", nframes, "with cropped frame dimensions: ",
          clip.size)
    print("Generating frames")
    for index in tqdm(range(nframes)):

        imagename = tmpfolder + "/file%04d.png" % index
        if os.path.isfile(tmpfolder + "/file%04d.png" % index):
            continue

        plt.axis('off')
        image = img_as_ubyte(clip.reader.read_frame())

        if np.ndim(image) > 2:
            h, w, nc = np.shape(image)
        else:
            h, w = np.shape(image)

        plt.figure(frameon=False, figsize=(w * 1. / 100, h * 1. / 100))
        plt.subplots_adjust(left=0,
                            bottom=0,
                            right=1,
                            top=1,
                            wspace=0,
                            hspace=0)
        plt.imshow(image)

        for bpindex, bp in enumerate(bodyparts2plot):
            if (Dataframe[scorer][bp]['likelihood'].values[index] >
                    CONF.evaluation.pcutoff):
                plt.scatter(
                    Dataframe[scorer][bp]['x'].values[index],
                    Dataframe[scorer][bp]['y'].values[index],
                    s=CONF.label.label_size**2,
                    color=colors(bpindex),
                    alpha=CONF.label.alpha)

        plt.xlim(0, w)
        plt.ylim(0, h)
        plt.axis('off')
        plt.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.gca().invert_yaxis()
        plt.savefig(imagename)

        plt.close("all")

    print("Generating video")

    subprocess.call([
        'ffmpeg',
        '-framerate', str(clip.fps),
        '-i', os.path.join(tmpfolder, 'file%04d.png'),
        '-r', '30',
        os.path.join(outdir, vname + '_DeepLabCutlabeled.mp4')])


def main(videofolder=None):
    myconfig.parse_args()

    # Name for scorer based on passed on parameters from myconfig_analysis.
    # Make sure they refer to the network of interest.
    scorer = paths.get_scorer_name(CONF.net.resnet,
                                   CONF.analysis.trainings_fraction,
                                   CONF.analysis.shuffle_index,
                                   CONF.analysis.trainings_iterations)

    ##################################################
    # Datafolder
    ##################################################

    if videofolder is None:
        videofolder = paths.get_video_dir()

    videos = paths.get_videos(videofolder)
    print("Starting ", videos)
    for video in videos:
        vname = os.path.basename(video.split('.')[0])

        outdir = paths.get_video_outdir(video)

        if os.path.isfile(os.path.join(outdir,
                                       vname + '_DeepLabCutlabeled.mp4')):
            print("Labeled video already created.")
        else:
            print("Loading ", video, "and data.")
            dataname = paths.get_video_dataname(video, scorer)
            # to load data for this video + scorer

            # FIXME(aloga): use correct tmp folder here
            tmpfolder = tempfile.mkdtemp(prefix="tmp_" + vname,
                                         dir=outdir)
            try:
                Dataframe = pd.read_hdf(dataname)
                clip = VideoFileClip(os.path.join(videofolder, video))
                CreateVideo(clip, Dataframe, videofolder, tmpfolder, vname)
            except FileNotFoundError:
                datanames = paths.get_video_all_datanames(video)
                if len(datanames) == 0:
                    print("The video was not analyzed with this scorer:",
                          scorer)
                    print("No other scorers were found, please run "
                          "AnalysisVideos.py first.")
                elif len(datanames) > 0:
                    print("The video was not analyzed with this scorer:",
                          scorer)
                    print("Other scorers were found, however:",
                          datanames)
                    print("Creating labeled video for:",
                          datanames[0],
                          " instead.")

                    Dataframe = pd.read_hdf(datanames[0])
                    clip = VideoFileClip(video)
                    CreateVideo(clip, Dataframe, outdir, tmpfolder, vname)
            finally:
                if CONF.analysis.delete_individual_frames:
                    shutil.rmtree(tmpfolder, ignore_errors=True)


if __name__ == "__main__":
    main()
