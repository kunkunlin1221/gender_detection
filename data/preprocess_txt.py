from random import shuffle

import capybara as cb
from fire import Fire


def main(folder):
    files = cb.get_files(folder, suffix=".jpg")
    folder = cb.Path(folder).absolute()
    shuffle(files)
    txt_for_test = []
    txt_for_train = []
    for i, file in enumerate(files):
        file = file.relative_to(folder)
        label = 0 if file.parent.name == "man" else 1
        if i < 1000:
            txt_for_test.append(f"{file} {label}\n")
        else:
            txt_for_train.append(f"{file} {label}\n")

    with open(folder / "train.txt", "w") as f:
        f.writelines(txt_for_train)

    with open(folder / "test.txt", "w") as f:
        f.writelines(txt_for_test)


Fire(main)
