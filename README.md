# YOLO11 Person Tracking

A Python script for tracking people in videos using YOLO11 and BoTSORT tracker.

## Features

- Person detection and tracking using YOLO11 (large model)
- BoTSORT tracker for stable track IDs
- Custom video annotation with track IDs
- CSV export of tracking data (personId, frame_idx, x, y)
- Optimized for detecting many people in crowded scenes

## Requirements

- Python 3.8+
- ultralytics
- opencv-python

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd yolo11-track
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download YOLO11 model (will be downloaded automatically on first run):
   - The script uses `yolo11l.pt` by default
   - Models are automatically downloaded from Ultralytics on first use

## Usage

### Basic Usage

Track people in a video with default settings:
```bash
python track_people.py
```

This will:
- Process `Lakeside.mp4` (default video)
- Use `yolo11l.pt` model
- Use `botsort.yaml` tracker
- Save results to `tracks.csv`
- Save annotated video to `runs/detect/track/Lakeside_tracked.mp4`

### Custom Video

Process a different video:
```bash
python track_people.py --video path/to/your/video.mp4
```

### Custom Output

Specify custom output files:
```bash
python track_people.py --video input.mp4 --output my_tracks.csv
```

### Advanced Options

```bash
python track_people.py \
  --video input.mp4 \
  --output tracks.csv \
  --model yolo11l.pt \
  --tracker botsort.yaml \
  --conf 0.05 \
  --imgsz 2560 \
  --max-det 500 \
  --font-size 0.3
```

#### Arguments

- `--video`: Input video path (default: `Lakeside.mp4`)
- `--output`: Output CSV file path (default: `tracks.csv`)
- `--model`: YOLO model to use (default: `yolo11l.pt`)
- `--tracker`: Tracker to use (default: `botsort.yaml`)
- `--conf`: Detection confidence threshold, lower = more detections (default: `0.05`)
- `--imgsz`: Image size for inference, larger = better for small/distant objects (default: `2560`)
- `--max-det`: Maximum detections per frame (default: `500`)
- `--font-size`: Font size for video labels (default: `0.3`)

## Output Format

### CSV File

The CSV file (`tracks.csv`) contains tracking data with the following columns:
- `personId`: Unique track ID for each person
- `frame_idx`: Frame number (0-indexed)
- `x`: X coordinate of person center (pixels)
- `y`: Y coordinate of person center (pixels)

Example:
```csv
personId,frame_idx,x,y
1,0,720.5,1280.3
1,1,722.1,1281.0
2,0,500.2,600.8
...
```

### Annotated Video

The annotated video shows:
- Bounding boxes around detected people
- Track ID numbers (small font)
- Saved to `runs/detect/track/<video_name>_tracked.mp4`

## Configuration

The script uses optimized default settings for detecting many people:
- **Model**: `yolo11l.pt` (large model for better accuracy)
- **Tracker**: `botsort.yaml` (BoTSORT for stable tracking)
- **Image Size**: `2560` (high resolution for small/distant objects)
- **Confidence**: `0.05` (low threshold to catch more detections)
- **Max Detections**: `500` (high limit for crowded scenes)

## Notes

- Model files (`.pt`) are automatically downloaded on first use
- Processing time depends on video length and resolution
- The script processes the entire video by default
- Output files are saved in the `runs/detect/track/` directory

## License

[Add your license here]

