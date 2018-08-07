from deeplabcut import myconfig
from deeplabcut.data import labels


def main():
    myconfig.parse_args()
    return labels.convert_labels_to_data_frame()


if __name__ == "__main__":
    main()
