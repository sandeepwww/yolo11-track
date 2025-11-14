import argparse
import csv
import cv2
import os
import tempfile
import yaml
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Parse command line arguments
parser = argparse.ArgumentParser(description='Track people in video and save to CSV')
parser.add_argument('--video', type=str, default='Lakeside.mp4', help='Input video path')
parser.add_argument('--output', type=str, default='tracks.csv', help='Output CSV file path')
parser.add_argument('--model', type=str, default='yolo11l.pt', help='YOLO model to use')
parser.add_argument('--tracker', type=str, default='botsort.yaml', help='Tracker to use')
parser.add_argument('--conf', type=float, default=0.05, help='Detection confidence threshold')
parser.add_argument('--imgsz', type=int, default=2560, help='Image size for inference')
parser.add_argument('--max-det', type=int, default=500, help='Maximum detections per frame')
parser.add_argument('--font-size', type=float, default=0.3, help='Font size for labels')
args = parser.parse_args()

VIDEO_PATH = args.video
OUTPUT_CSV = args.output

# Load model
model = YOLO(args.model)

# Run tracking
results = model.track(
    source=VIDEO_PATH,
    tracker=args.tracker,
    stream=True,
    persist=True,
    conf=args.conf,
    iou=0.4,
    imgsz=args.imgsz,
    classes=[0],
    agnostic_nms=False,
    max_det=args.max_det,
    save=False,
)

# Get video properties for output video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Create output video path
output_dir = Path('runs/detect/track')
output_dir.mkdir(parents=True, exist_ok=True)
video_name = Path(VIDEO_PATH).stem
output_video_path = output_dir / f"{video_name}_tracked.mp4"

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

# Process frames and write to CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["personId", "frame_idx", "x", "y"])
    
    frame_idx = 0
    
    for result in results:
        im0 = result.orig_img.copy() if hasattr(result, 'orig_img') else result.plot()
        annotator = Annotator(im0, line_width=2, font_size=args.font_size)
        
        if result.boxes is not None and result.boxes.is_track:
            classes = result.boxes.cls.cpu().tolist()
            boxes = result.boxes.xyxy.cpu().tolist()
            boxes_xywh = result.boxes.xywh.cpu().tolist()
            ids = result.boxes.id.int().cpu().tolist()
            
            for cls, box_xyxy, (x_center, y_center, w, h), tid in zip(classes, boxes, boxes_xywh, ids):
                if int(cls) == 0:  # Only person class
                    writer.writerow([tid, frame_idx, x_center, y_center])
                    label = str(int(tid))
                    annotator.box_label(box_xyxy, label, color=colors(int(tid), True))
        
        out_video.write(im0)
        frame_idx += 1

# Close video writer
out_video.release()

print(f"Processing complete!")
print(f"Total frames processed: {frame_idx}")
print(f"Saved tracks to {OUTPUT_CSV}")
print(f"Saved annotated video to {output_video_path}")
