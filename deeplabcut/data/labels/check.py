from deeplabcut import myconfig
from deeplabcut.data import labels


def main():
    myconfig.parse_args()
    return labels.check_labels()


if __name__ == "__main__":
    main()
