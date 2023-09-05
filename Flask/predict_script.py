from flask import Flask, render_template, request, redirect, url_for
from gunicorn.app.base import Application
from werkzeug.utils import secure_filename
import torch
import ultralytics
from ultralytics import YOLO
import os,sys
import subprocess

def subprocess_predict(model_name):

    print("Starting model inference...")
    path = 'datasets/sample/model/' + model_name + '.pt'
    model = YOLO(path)
    # frame = "datasets/sample/t1.jpg"
    frame = "datasets/sample/example.mp4"
    results = model.predict(frame,
                            conf=0.1,
                            iou=0.7,
                            agnostic_nms=True,
                            device='cpu',
                            stream=False
                            )
    print("Model inference completed!")
    return 1

if __name__ == "__main__":
    model_name = sys.argv[1]  # Get model path from command-line argument
    subprocess_predict(model_name)