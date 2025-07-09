#!/bin/bash

ultralytics export model=./models/custom.pt format=engine device=0 imgsz=320 half=True
mv ./models/custom.engine ./models/custom_320_FP16.engine

ultralytics export model=./models/NubsUpDown.pt format=engine device=0 imgsz=320 half=True
mv ./models/NubsUpDown.engine ./models/NubsUpDown_320_FP16.engine

ultralytics export model=./models/synthetic.pt format=engine device=0 imgsz=320 half=True
mv ./models/synthetic.engine ./models/synthetic_320_FP16.engine