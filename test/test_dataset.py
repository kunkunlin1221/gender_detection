import capybara as cb
from fire import Fire

from src.dataset import GenderDataset


def main(txt_file):
    dataset = GenderDataset(txt_file)
    for i in range(len(dataset)):
        img, label = dataset[i]
        img = dataset.restore_img(img)
        cb.imwrite(img)


if __name__ == "__main__":
    Fire(main)
