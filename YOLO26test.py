from ultralytics import YOLO
model = YOLO('yolo26n.pt')
model.predict(source='WhatsApp Image 2026-02-03 at 10.36.32 (1).jpeg', conf=0.25, save=True, show=True)