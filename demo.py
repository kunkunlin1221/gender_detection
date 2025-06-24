from functools import partial

import capybara as cb
import facekit as fk
import numpy as np
from fire import Fire

from src.dataset import imresize_and_pad_if_need


class GenderDetection:
    def __init__(self, onnx_path):
        self.engine = cb.ONNXEngine(onnx_path, backend="cpu")
        self.mean = np.array([0.485, 0.456, 0.406], dtype="float32")
        self.std = np.array([0.229, 0.224, 0.225], dtype="float32")

    def __call__(self, x):
        x = imresize_and_pad_if_need(x, 112, 112)
        x = x.astype("float32") / 255.0
        x = (x - self.mean) / self.std
        blob = np.transpose(x, (2, 0, 1))[None]  # HWC to CHW
        output = self.engine(input=blob)["output"]
        return "female" if output.argmax(axis=1).item() else "male"


def pipeline(img, face_service, gender_detection):
    faces = face_service([img])[0]
    if len(faces):
        face = faces[0]
        cropped = cb.imcropbox(img, face.box)
        gender = gender_detection(cropped)
        img = cb.draw_box(img, face.box)
        img = cb.draw_text(img, gender, location=face.box.left_top, color=(0, 255, 0), text_size=20)
    return img


def main(camera_ip=0, onnx_path="gender_detection_lcnet_050.onnx"):
    face_service = fk.FaceService()
    gender_detection = GenderDetection(onnx_path)
    pipeline2 = partial(pipeline, face_service=face_service, gender_detection=gender_detection)
    demo = cb.WebDemo(camera_ip, pipelines=[pipeline2])
    demo.run()


Fire(main)
