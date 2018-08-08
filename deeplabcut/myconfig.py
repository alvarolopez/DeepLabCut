# coding: utf-8

import sys

from oslo_config import cfg

CONF = cfg.CONF

# General options

opts = [
    cfg.StrOpt(
        "task",
        default="reaching",
        help="""
Task that is being performed in the video. This is used for indentifying a
        trained model that is specialized in this task. We will use this string
        as a label for all the steps of the DeepLabCut toolbox.
"""
    ),
    cfg.StrOpt(
        "base-directory",
        default="data",
        help="""
Base directory for data. All the data that will be read and written will be
        done within this directory. You can check which structure we are using
        by issuing the command `dlc-config-print-paths`.
"""
    ),
]

CONF.register_cli_opts(opts)

# Data sampling options
# FIXME(aloga): hide these when not issuing any related command
sampling_opts = [
    cfg.StrOpt(
        "video-file",
        default="reachingvideo1.avi",
        help="""
Video file to extract frames from. The video should be stored in the
        ${base_directory}/raw/${task} directory. We will extract the frames
        into the  ${base_directory}/frames/${task}/ directory.
"""
    ),
    cfg.IntOpt(
        "portion",
        default=1,
        help="""
Portion of the video to sample from in step 1. Set to 1 by default.
"""
    ),
]
CONF.register_opts(sampling_opts, group="sampling")

# Common video options, in theory these should be used when sampling and when
# making the analysis.
video_opts = [
    cfg.BoolOpt(
        "cropping",
        default=True,
        help="""
Whether we should crop the video or not, taking into account the x1, x2, y1, y2
        coordinates.
"""
    ),
    cfg.IntOpt(
        "x1",
        default=0,
        help="""
ROI dimensions / bounding box (only used if cropping == True) x1,y1 indicates
        the top left corner and x2,y2 is the lower right corner of the croped
        region.
"""),
    cfg.IntOpt(
        "x2",
        default=640,
        help="""
ROI dimensions / bounding box (only used if cropping == True) x1,y1 indicates
        the top left corner and x2,y2 is the lower right corner of the croped
        region.
"""),
    cfg.IntOpt(
        "y1",
        default=277,
        help="""
ROI dimensions / bounding box (only used if cropping == True) x1,y1 indicates
        the top left corner and x2,y2 is the lower right corner of the croped
        region.
"""),
    cfg.IntOpt(
        "y2",
        default=624,
        help="""
ROI dimensions / bounding box (only used if cropping == True) x1,y1 indicates
        the top left corner and x2,y2 is the lower right corner of the croped
        region.
"""),
]

CONF.register_opts(video_opts, group="video")

# Options that are related with the human labelling of the images, that is, who
# did the labelling and how this is stored.
labelling_opts = [
    cfg.ListOpt(
        "bodyparts",
        default=["hand", "Finger1", "Finger2", "Joystick"],
        help="""
Exact sequence of labels as were put by annotator in *csv files
"""
    ),
    cfg.StrOpt(
        "scorer",
        default="Mackenzie",
        help="""
Who did the labelling of the data.
"""
    ),
    cfg.BoolOpt(
        "multibodypartsfile",
        default=False,
        help="""
Set this true if the data was sequentially labeled and if there is one file per
        folder (you can set the name of this file).  Otherwise there should be
        individual files per bodypart, i.e. in our demo case hand.csv,
        Finger1.csv etc.  If true then those files will be generated from
        Results.txt
"""
    ),
    cfg.StrOpt(
        "multibodypartsfilename",
        default="Results.csv",
        help="""
File name to use when multibodypartsfile is set to True.
"""
    ),
    cfg.IntOpt(
        "invisibleboundary",
        default=10,
        help="""
When importing the images and the labels in the csv/xls files should be in the
        same order!  During labeling in Fiji one can thus (for occluded body
        parts) click in the origin of the image i.e. top left corner (close to
        0,0)) these 'false' labels will then be removed. To do so set the
        following variable: set this to 0 if no labels should be removed!" If
        labels are closer to origin than this number they are set to NaN.
        Please adjust to your situation. Units in pixel.
"""
    ),
    cfg.StrOpt(
        "imagetype",
        default=".png",
        help="""
Image of extracted frames, defaults to ".png".  If you started from already
        extracted frames in a different format then change the format here.
"""
    ),
]

CONF.register_opts(labelling_opts, group="labelling")

# Options related with how the labels will be plotted
label_opts = [
    cfg.StrOpt(
        "label",
        default=".",
        help="""
Matplotlib label to use (e.g. ".", "+", "*") for the labelling (plotting) of
        the images.
"""
    ),
    cfg.StrOpt(
        "colormap",
        default="cool",
        help="""
Matplotlib color map (e.g. viridis, cool, hsv) to use for the labelling
        (ploting) of images.
"""
    ),
    cfg.FloatOpt(
        "scale",
        default=1,
        help="""
Matplotlib scaling for the labelling (plotting) of the images.
"""
    ),
    cfg.IntOpt(
        "label-size",
        default=10,
        help="""
Label size for the labelling (plotting) of the images.
"""
    ),
    cfg.FloatOpt(
        "alpha",
        default=0.6,
        help="""
Label transparency level for the labelling (plotting) of the images.
"""
    ),
]

CONF.register_opts(label_opts, group="label")

########################################
# Step 4: Generate Training Files
########################################

net_opts = [
    cfg.StrOpt(
        "date",
        default="Jan30",
        help="""
Date when the training was done. This is just used to identify the network
        configuration.
"""
    ),
    cfg.ListOpt(
        "shuffles",
        default=[1],
        item_type=int,
        help="""
Identificators for all the shuffles that we will use for traning.
"""
    ),
    cfg.ListOpt(
        "training-fraction",
        default=[0.95],
        item_type=float,
        help="""
Fraction of labeled images used for training.
"""
    ),
    cfg.StrOpt(
        "resnet",
        default="50",
        choices=["101", "50"],
        help="""
Which resnet to use, valid values are "101" or "50", use "50" by default.
"""
    ),
]
CONF.register_opts(net_opts, group="net")

evaluation_opts = [
    cfg.StrOpt(
        "snapshotindex",
        default="-1",
        help="""
To evaluate the last model that was trained most set this to: -1, to evaluate
        all models (training stages) set this to: 'all'  (as string!)
"""
    ),
    cfg.BoolOpt(
        "plotting",
        default=True,
        help="""
If true will plot train & test images including DeepLabCut labels next to human
        labels. Note that this will be plotted for all snapshots as indicated
        by snapshotindex
"""
    ),
    cfg.FloatOpt(
        "pcutoff",
        default=.1,
        help="""
likelihood. RMSE will be reported for all pairs and pairs with larger
        likelihood than pcutoff (see paper). This cutoff will also be used in
        plots.
"""
    ),
]
CONF.register_opts(evaluation_opts, group="evaluation")

# Analysis options

analysis_opts = [
    cfg.StrOpt(
        "video-directory",
        default="${base_directory}/videos",
        help="""
Base directory for videos to analyze.
"""
    ),
    cfg.StrOpt(
        "video-type",
        default=".avi",
        help="""
Type of videos to analyze
"""
    ),
    cfg.IntOpt(
        "trainings-iterations",
        default=500,
        help="""
Type the number listed in the h5 file containing the pose estimation data. The
        video will be generated
"""
    ),
    cfg.FloatOpt(
        "trainings-fraction",
        default=0.95,
        help="""
Fraction of labeled images used for training
"""
    ),
    cfg.IntOpt(
        "shuffle-index",
        default=1,
    ),
    cfg.IntOpt(
        "snapshot-index",
        default=-1,
    ),
    cfg.BoolOpt(
        "store-as-csv",
        default=False,
    ),
    cfg.BoolOpt(
        "delete-individual-frames",
        default=False,
    ),
]

CONF.register_opts(analysis_opts, group="analysis")


def list_opts():
    return [
        ("DEFAULT", opts),
        ("sampling", sampling_opts),
        ("video", video_opts),
        ("labelling", labelling_opts),
        ("label", label_opts),
        ("net", net_opts),
        ("evaluation", evaluation_opts),
        ("analysis", analysis_opts),
    ]


def parse_args(argv=sys.argv, default_config_files=None):
    CONF(argv[1:],
         project='dlc',
         default_config_files=default_config_files)
