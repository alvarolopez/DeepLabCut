"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut

A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

"""

import os
import math

import imageio
import matplotlib
from moviepy.editor import VideoFileClip
from skimage import io
from skimage.util import img_as_ubyte
import numpy as np

from deeplabcut import myconfig
from deeplabcut import paths
from deeplabcut import utils

CONF = myconfig.CONF

imageio.plugins.ffmpeg.download()
matplotlib.use('Agg')


def select_random_frames(task=CONF.data.task):
    """Select random frames from a video.

    A key point is to select diverse frames, which are typical for the behavior
    you study that should be labeled. This helper script selects N frames
    uniformly sampled from a particular video. Ideally you would also get data
    from different session and different animals if those vary substantially.
    Note: this might not yield diverse frames, if the behavior is sparsely
    distributed.

    Individual images should not be too big (i.e. < 850 x 850 pixel). Although
    this can be taken care of later as well, it is advisable to crop the
    frames, to remove unnecessary parts of the video as much as possible.
    """

    frame_folder = paths.get_frame_dir()
    utils.attempttomakefolder(frame_folder)

    #####################################################################
    # First load the image and crop (if necessary).
    #####################################################################

    # Number of frames to pick (set this to 0 until you found right cropping)
    numframes2pick = 10

    video_file = paths.get_raw_video_file()
    clip = VideoFileClip(video_file)
    print("Duration of video [s], ", clip.duration, "fps, ", clip.fps,
          "Cropped frame dimensions: ", clip.size)

    ny, nx = clip.size  # dimensions of frame (width, height)

    # Select ROI of interest by adjusting values in myconfig.py
    # USAGE:
    # clip.crop(x1=None, y1=None, x2=None, y2=None, width=None, height=None,
    # x_center=None, y_center=None)
    #
    # Returns a new clip in which just a rectangular subregion of the
    # original clip is conserved. x1,y1 indicates the top left corner and
    # x2,y2 is the lower right corner of the croped region.
    #
    # All coordinates are in pixels. Float numbers are accepted.
    if CONF.data.cropping:
        clip = clip.crop(y1=CONF.data.y1,
                         y2=CONF.data.y2,
                         x1=CONF.data.x1,
                         x2=CONF.data.x2)

    image = clip.get_frame(1.2)
    imgname = os.path.join(frame_folder, "IsCroppingOK.png")
    io.imsave(imgname, image)
    print("--> Open %s file to set the output range! <---" % imgname)
    print("--> Adjust shiftx, shifty, fx and fy accordingly! <---")

    ####################################################
    # Creating folder with name of experiment and extract random frames
    ####################################################

    print("Videoname: ", CONF.data.video_file)
    folder = paths.get_video_frames_dir()
    print(folder)
    utils.attempttomakefolder(folder)

    frames = np.random.randint(
        math.floor(clip.duration * clip.fps * CONF.data.portion),
        size=numframes2pick - 1
    )
    width = int(np.ceil(np.log10(clip.duration * clip.fps)))

    for index in frames:
        try:
            image = img_as_ubyte(clip.get_frame(index * 1. / clip.fps))
            imgname = os.path.join(folder,
                                   "img" + str(index).zfill(width) + ".png")
            io.imsave(imgname, image)
        except FileNotFoundError:
            print("Frame # ", index, " does not exist.")

    # Extract the first frame (not cropped!) - useful for data augmentation
    clip = VideoFileClip(video_file)
    index = 0
    image = img_as_ubyte(clip.get_frame(index * 1. / clip.fps))
    imgname = os.path.join(folder, "img" + str(index).zfill(width) + ".png")
    io.imsave(imgname, image)


def main():
    myconfig.parse_args()
    return select_random_frames()


if __name__ == "__main__":
    main()
