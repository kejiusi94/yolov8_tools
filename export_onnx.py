from ultralytics import YOLO

model = YOLO(r"you_modeL_path")

success = model.export(format="onnx")


