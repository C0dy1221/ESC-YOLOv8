from ultralytics.models.yolo.segment import SegmentationValidator

args = dict(model='D:/EMA-yolov8-SPPLEAN/runs/segment/train7/weights/best.pt', data='D:/EMA-yolov8-SPPLEAN/0segdatasetsmaker/sickfish_real.yaml')
validator = SegmentationValidator(args=args)
validator()