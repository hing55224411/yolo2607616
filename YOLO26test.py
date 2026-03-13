from ultralytics import YOLO
model = YOLO('best (1).pt')
model.predict(source='0. Star Photo', conf=0.25, save=True, show=True)