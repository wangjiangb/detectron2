
from detectron2.utils.file_io import PathManager


def load_sem_seg_pairs(pair_path):
    image_files = []
    label_files = []
    with PathManager.open(pair_path, 'r') as f:
        lines = f.read().split("\n")
        lines = [line.strip() for line in lines if len(line)]
        lines = [line.split(", ") for line in lines]
    ret = []

    for line in lines:
        image_file, label_file, h, w  = line
        h = int(h)
        w = int(w)
        ret.append(
            {
                "file_name": image_file,
                "sem_seg_file_name": label_file,
                "height": h,
                "width": w,
            }
        )
        image_files.append(image_file)
        label_files.append(label_file)

    assert len(ret), "No image is available"
    return ret


if __name__ == '__main__':
    print(len(load_sem_seg_pairs("/home/jiangwang/code/Human-Segmentation-PyTorch/dataset/all_train_mask.txt")))
