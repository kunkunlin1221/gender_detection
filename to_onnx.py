import capybara as cb
import torch
from fire import Fire

from src.lightning import LightningModule


def main(ckpt_path):
    model = LightningModule.load_from_checkpoint(ckpt_path)
    model.example_input_array = torch.randn(1, 3, 112, 112)
    onnx_fpath = cb.Path(ckpt_path).with_suffix(".onnx")
    model.to_onnx(onnx_fpath, export_params=True, input_names=["input"], output_names=["output"])


if __name__ == "__main__":
    Fire(main)
