import os
import shutil


def split_dataset(src: str, dst: str, rate: tuple, num: int):
    total_rate = sum(rate)
    train_num = rate[0] / total_rate * num
    val_num = rate[1] / total_rate * num
    for idx, file in enumerate(os.listdir(src)):
        if idx < train_num:
            shutil.copy(src + file, dst + "train/" + file)
        elif idx < train_num + val_num:
            shutil.copy(src + file, dst + "val/" + file)
        elif idx < num:
            shutil.copy(src + file, dst + "test/" + file)
        else:
            break


if __name__ == "__main__":
    split_dataset("/Users/keigatsu/Downloads/train/", "data/", (8, 1, 1), 100)
