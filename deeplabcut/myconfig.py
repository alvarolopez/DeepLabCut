# coding: utf-8

from oslo_config import cfg

CONF = cfg.CONF

data_opts = [
        cfg.StrOpt(
            "task",
            default="reaching",
            help="Task that is being performed in the video."
        ),
        cfg.StrOpt(
            "base-directory",
            default="data",
            help="Base directory for data."
        ),
        cfg.StrOpt(
            "video-file",
            default="reachingvideo1.avi",
            help="Video to extract frames from."
        ),
        cfg.BoolOpt(
            "cropping",
            default=True,
        ),
        cfg.IntOpt(
            "x1",
            default=0,
            help=("ROI dimensions / bounding box (only used if cropping == True)"
                  "x1,y1 indicates the top left corner and"
                  "x2,y2 is the lower right corner of the croped region.")
        ),
        cfg.IntOpt(
            "x2",
            default=640,
            help=("ROI dimensions / bounding box (only used if cropping == True)"
                  "x1,y1 indicates the top left corner and"
                  "x2,y2 is the lower right corner of the croped region.")
        ),
        cfg.IntOpt(
            "y1",
            default=277,
            help=("ROI dimensions / bounding box (only used if cropping == True)"
                  "x1,y1 indicates the top left corner and"
                  "x2,y2 is the lower right corner of the croped region.")
        ),
        cfg.IntOpt(
            "y2",
            default=624,
            help=("ROI dimensions / bounding box (only used if cropping == True)"
                  "x1,y1 indicates the top left corner and"
                  "x2,y2 is the lower right corner of the croped region.")
        ),
        cfg.IntOpt(
            "portion",
            default=1,
            help=("Portion of the video to sample from in step 1. Set to 1 by default.")
        ),
]

CONF.register_cli_opts(data_opts, group="data")

dataframe_opts = [
    cfg.ListOpt("bodyparts",
                default=["hand", "Finger1", "Finger2","Joystick"],
                help="Exact sequence of labels as were put by "
                     "annotator in *csv files"
    ),
    cfg.ListOpt("scorers",
                default=["Mackenzie"],
                help="Who is labelling the data"
    ),
    cfg.BoolOpt("multibodypartsfile",
                default=False,
                help="Set this true if the data was sequentially "
                     "labeled and if there is one file per folder "
                     "(you can set the name of this file)."
                     "Otherwise there should be individual files per "
                     "bodypart, i.e. in our demo case hand.csv, "
                     "Finger1.csv etc."
                     "If true then those files will be generated from "
                     "Results.txt"
    ),
    cfg.StrOpt("multibodypartsfilename",
               default="Results.csv",
               help="File name to use when multibodypartsfile=True"
    ),
    cfg.IntOpt("invisibleboundary",
               default=10,
               help="When importing the images and the labels in the csv/xls "
                    "files should be in the same order!  During labeling in "
                    "Fiji one can thus (for occluded body parts) click in the "
                    "origin of the image i.e. top left corner (close to 0,0)),"
                    " these 'false' labels will then be removed. To do so set "
                    "the following variable: set this to 0 if no labels should"
                    " be removed!"
                    "If labels are closer to origin than this number they are "
                    "set to NaN. Please adjust to your situation."
                    "Units in pixel."
    ),
    cfg.StrOpt("imagetype",
               default=".png",
               help="Image type of extracted frames (do not change if you used"
                    " our step1). If you started from already extracted frames"
                    "in a different format then change the format here (for "
                    "step2 to 4)."
    ),
]

CONF.register_cli_opts(dataframe_opts, group="dataframe")

#####################################################################################
# This configuration file sets various parameters for generation of training
# set & evalutation of DeepLabCut
#####################################################################################

# myconfig.py:

########################################
# Step 1: Selecting Frames from videos
########################################

#Task = 'reaching'

# Filename and path to behavioral video:
#vidpath = '.'
#filename = 'reachingvideo1.avi'

#cropping = True

# ROI dimensions / bounding box (only used if cropping == True)
# x1,y1 indicates the top left corner and
# x2,y2 is the lower right corner of the croped region.

#x1 = 0
#x2 = 640
#y1 = 277
#y2 = 624

# Portion of the video to sample from in step 1. Set to 1 by default.
#portion = 1

########################################
# Step 2: Converting frames to pandas array
########################################

#bodyparts = ["hand", "Finger1", "Finger2","Joystick"]  # Exact sequence of labels as were put by
# annotator in *.csv file
#Scorers = ['Mackenzie']  # who is labeling?

# Set this true if the data was sequentially labeled and if there is one file per folder (you can set the name of this file below, i.e. multibodypartsfilename)
# Otherwise there should be individual files per bodypart, i.e. in our demo case hand.csv, Finger1.csv etc.
# If true then those files will be generated from Results.txt
#multibodypartsfile=False
#multibodypartsfilename="Results.csv"

# When importing the images and the labels in the csv/xls files should be in the same order!
# During labeling in Fiji one can thus (for occluded body parts) click in the origin of the image
#(i.e. top left corner (close to 0,0)), these "false" labels will then be removed. To do so set the following variable:
#set this to 0 if no labels should be removed!

#invisibleboundary=10 # If labels are closer to origin than this number they are set to NaN (not a number). Please adjust to your situation. Units in pixel.

#imagetype=".png" # image type of extracted frames (do not change if you used our step1).
# If you started from already extracted frames in a different format then change the format here (for step2 to 4).

########################################
# Step 3: Check labels / makes plots
########################################

label_opts = [
    cfg.StrOpt("colormap",
               default="cool",
               help="Set color map (e.g. viridis, cool, hsv)"
    ),
    cfg.FloatOpt("scale",
                 default=1,
                 help="Set scaling for plotting"
    ),
    cfg.IntOpt("label-size",
               default=10,
               help="Label size"
    ),
    cfg.FloatOpt("alpha",
                 default=0.6,
                 help="Label transparency"
    ),
    cfg.StrOpt("scorer",
               default="Mackenzie",
               help="Who has labelled the data"
    ),
]

CONF.register_opts(label_opts, group="label")
#colormap = 'cool' #set color map, i.e. viridis, cool, hsv
#scale = 1  # for plotting
#msize=10   #size of labels
#alphavalue =.6 #transparency of labels

########################################
# Step 4: Generate Training Files
########################################

net_opts = [
    cfg.StrOpt("date",
               default="Jan30",
    ),
    cfg.ListOpt("shuffles",
                default=[1],
                item_type=int,
                help="Ids for shuffles, i.e. range(5) for 5 shuffles"
    ),
    cfg.ListOpt("training-fraction",
                default=[0.95],
                item_type=float,
                help="Fraction of labeled images used for training"
    ),
    cfg.StrOpt("resnet",
               default="50",
               help="Which resnet to use",
    ),
]
CONF.register_opts(net_opts, group="net")
#date = 'Jan30'
#scorer = 'Mackenzie'

evaluation_opts = [
    cfg.StrOpt("snapshotindex",
               default="-1",
               help="To evaluate the last model that was trained most set this to: -1, "
                    "To evaluate all models (training stages) set this to: 'all'  (as string!)"
    ),
]
CONF.register_opts(evaluation_opts, group="evaluation")
# Userparameters for training set. Other parameters can be set in pose_cfg.yaml
#Shuffles = [1]  # Ids for shuffles, i.e. range(5) for 5 shuffles
#TrainingFraction = [0.95]  # Fraction of labeled images used for training

# Which resnet to use
# (these are parameters reflected in the pose_cfg.yaml file)
#resnet = 50

# For Evaluation/ Analyzing videos
# To evaluate the last model that was trained most set this to: -1
# To evaluate all models (training stages) set this to: "all"  (as string!)

#snapshotindex = -1 #"all"
#shuffleindex = 0
#pcutoff=.1 # likelihood. RMSE will be reported for all pairs and pairs with larger likelihood than pcutoff (see paper). This cutoff will also be used in plots.
#plotting=True #If true will plot train & test images including DeepLabCut labels next to human labels. Note that this will be plotted for all snapshots as indicated by snapshotindex
