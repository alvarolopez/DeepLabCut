import os.path

from deeplabcut import myconfig

CONF = myconfig.CONF

base_dir = os.path.abspath(CONF.data.base_directory)
frame_dir = os.path.join(base_dir, "frames", CONF.data.task)
raw_dir = os.path.join(base_dir, "raw", CONF.data.task)
label_dir = os.path.join(base_dir, "labels", CONF.data.task)
tmp_dir = os.path.join(base_dir, "tmp", CONF.data.task)
train_dir = os.path.join(base_dir, "train", CONF.data.task)
pre_trained_dir = os.path.join(base_dir, "train", CONF.data.task, "pretrained")
results_dir = os.path.join(base_dir, "results", CONF.data.task)
video_dir = os.path.abspath(CONF.analysis.video_directory)
output_dir = os.path.join(base_dir, "output")


def get_raw_video_file():
    filename = CONF.data.video_file
    return os.path.join(raw_dir, filename)


def get_video_frames_dir():
    filename = CONF.data.video_file
    return os.path.join(frame_dir, filename.split('.')[0])


def get_tmp_dir(dirname):
    return os.path.join(tmp_dir, dirname)


def get_collected_data_file(scorer, filetype=".h5"):
    filename = 'CollectedData_' + scorer + filetype
    return os.path.join(label_dir, filename)


def get_video_datasets():
    folders = [
        videodatasets for videodatasets in os.listdir(frame_dir)
        if os.path.isdir(os.path.join(frame_dir, videodatasets))
    ]
    return folders


def get_video_dataset_frames(dataset):
    aux = os.path.join(frame_dir, dataset)
    files = [
        os.path.join(aux, fn) for fn in os.listdir(aux)
        if ("img" in fn and
            CONF.dataframe.imagetype in fn and
            "_labelled" not in fn)
    ]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return files


def get_train_dataset_dir():
    # Make that folder and put in the collecteddata (see below)
    bf = "UnaugmentedDataSet_" + CONF.data.task + CONF.net.date + "/"
    return os.path.join(train_dir, bf)


def get_train_matfile(train_fraction, shuffle):
    return os.path.join(
        get_train_dataset_dir(),
        CONF.data.task + "_" + CONF.label.scorer +
        str(int(100 * train_fraction)) + "shuffle" + str(shuffle) +
        ".mat"
    )


def get_train_docfile(train_fraction, shuffle):
    return os.path.join(
        get_train_dataset_dir(),
        "Documentation_" + CONF.data.task + "_" +
        str(int(train_fraction * 100)) + "shuffle" + str(shuffle) +
        ".pickle"
    )


def get_training_imagefile(filename):
    aux_path = os.path.relpath(filename, frame_dir)
    return os.path.abspath(os.path.join(get_train_dataset_dir(),
                                        "frames",
                                        aux_path))


def get_experiment_name(train_fraction, shuffle):
    return os.path.join(
        train_dir,
        CONF.data.task + CONF.net.date + '-trainset' +
        str(int(train_fraction * 100)) + 'shuffle' + str(shuffle)
    )


def get_pose_cfg_test(train_fraction, shuffle):
    return os.path.join(get_experiment_name(train_fraction, shuffle),
                        "test",
                        "pose_cfg.yaml")


def get_pose_cfg_train(train_fraction, shuffle):
    return os.path.join(get_experiment_name(train_fraction, shuffle),
                        "train",
                        "pose_cfg.yaml")


def get_pre_trained_file():
    return os.path.join(pre_trained_dir, "resnet_v1_50.ckpt")


def get_train_snapshots(trainFraction, shuffle):
    aux = os.path.join(get_experiment_name(trainFraction, shuffle), "train")
    return [os.path.join(aux, fn.split('.')[0])
            for fn in os.listdir(aux) if "index" in fn]


def get_scorer_name(net_type, train_fraction, shuffle, iters):
    return ('DeepCut' + "_resnet" + str(net_type) + "_" +
            str(int(train_fraction * 100)) + 'shuffle' + str(shuffle) +
            '_' + str(iters) + "forTask_" + CONF.data.task +
            str(CONF.net.date))


def get_scorer_file(net_type, train_fraction, shuffle, iters):
    scorer = get_scorer_name(net_type, train_fraction, shuffle, iters)
    return os.path.join(results_dir, scorer + ".h5")


def get_evaluation_files(train_fraction, shuffle):
    return [
        f for f in os.listdir(results_dir)
        if "forTask_" + str(CONF.data.task) in f and
        "shuffle" + str(shuffle) in f and
        "_" + str(int(train_fraction * 100)) in f
    ]


def get_videos(video_dir=video_dir):
    videotype = CONF.analysis.video_type
    return [os.path.join(video_dir, fn)
            for fn in os.listdir(video_dir) if (videotype in fn)]


def get_video_dataname(video, scorer):
    dataname = video.split('.')[0] + scorer + '.h5'
    # Get rid of prefix
    dataname = os.path.basename(dataname)
    return os.path.join(get_video_outdir(video), dataname)


def get_video_all_datanames(video):
    video = os.path.basename(video.split(".")[0])
    return [
        os.path.join(get_video_outdir(video), fn)
        for fn in os.listdir(get_video_outdir(video))
        if ((video in fn) and (".h5" in fn) and "resnet" in fn)
    ]


def get_video_outdir(video):
    video = os.path.basename(video)
    return os.path.join(output_dir,
                        video.rsplit(".", 1)[0])


def print_data_dirs():
    print("\t           base dir:", base_dir)
    print("\t     raw videos dir:", raw_dir)
    print("\t     raw video file:", get_raw_video_file())
    print("\t          frame dir:", frame_dir)
    print("\t   video frames dir:", get_video_frames_dir())
    print("\t         labels dir:", label_dir)
    for scorer in CONF.dataframe.scorers:
        print("\t        labels file:", get_collected_data_file(scorer=scorer))
    print("\t            tmp dir:", tmp_dir)
    print("\t          train dir:", train_dir)
    print("\t pre-trained TF dir:", pre_trained_dir)
    print("\t   pre-trained file:", get_pre_trained_file())
    print("\t        results dir:", results_dir)
    print("\t          video dir:", video_dir)
    print("\t         output dir:", output_dir)
