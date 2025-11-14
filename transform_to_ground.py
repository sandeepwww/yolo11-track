import csv
import cv2
import numpy as np
import argparse
from pathlib import Path

# Parse command line arguments
parser = argparse.ArgumentParser(description='Transform tracked coordinates to ground plane using homography')
parser.add_argument('--input', type=str, default='tracks.csv', help='Input CSV file with tracked coordinates')
parser.add_argument('--output', type=str, default='tracks_ground.csv', help='Output CSV file with ground plane coordinates')
parser.add_argument('--video-width', type=int, default=1920, help='Video frame width (default: 1920)')
parser.add_argument('--video-height', type=int, default=1080, help='Video frame height (default: 1080)')
parser.add_argument('--img-width', type=int, default=5996, help='Original image width for homography points (default: 5996)')
parser.add_argument('--img-height', type=int, default=3372, help='Original image height for homography points (default: 3372)')
args = parser.parse_args()

# Image points (in original image resolution: 5996x3372)
# These are in standard image coordinates with origin at top-left
img_pts = np.array([
    [5499, 3342],  # Point 1
    [4259, 1953],  # Point 2
    [2415, 792],   # Point 3
    [1515, 1002],  # Point 4
    [244, 1235],   # Point 5
], dtype=np.float32)

# World points (ground plane coordinates in meters, with bottom right vertex as [0,0])
world_pts_br = np.array([
    [0.00, 0.00],
    [2.67, 10.72],
    [0.65, 86.626],
    [-12.34, -54.30],
    [-10.07, -44.52],
], dtype=np.float32)

# Convert world coordinates from bottom-right origin to top-left origin
# to match the image coordinate system
# Find the maximum extent of world coordinates
max_x = np.max(world_pts_br[:, 0])
max_y = np.max(world_pts_br[:, 1])
# Convert: x_new = max_x - x_old, y_new = max_y - y_old
# This makes what was bottom-right [0,0] become [max_x, max_y] (top-left in new system)
world_pts = world_pts_br.copy()
world_pts[:, 0] = max_x - world_pts_br[:, 0]   # Flip x: right becomes left
world_pts[:, 1] = max_y - world_pts_br[:, 1]   # Flip y: bottom becomes top

print(f"Image points (top-left origin):")
print(img_pts)
print(f"\nWorld points (bottom-right origin):")
print(world_pts_br)
print(f"\nConverted world points (top-left origin):")
print(world_pts)

# Calculate homography matrix
# Using getPerspectiveTransform for 4 points or findHomography for 5+ points
if len(img_pts) == 4:
    H = cv2.getPerspectiveTransform(img_pts, world_pts)
else:
    # For 5+ points, use findHomography (more robust)
    # findHomography accepts points in (N, 2) or (N, 1, 2) format
    H, mask = cv2.findHomography(img_pts, world_pts, cv2.RANSAC, 5.0)
    if H is None:
        # If RANSAC fails, try without RANSAC
        print("Warning: RANSAC failed, trying without RANSAC...")
        H, mask = cv2.findHomography(img_pts, world_pts, 0)
    if H is None:
        # If that also fails, use first 4 points with getPerspectiveTransform
        print("Warning: findHomography failed, using first 4 points with getPerspectiveTransform...")
        H = cv2.getPerspectiveTransform(img_pts[:4], world_pts[:4])

if H is None:
    raise ValueError("Failed to calculate homography matrix. Please check your input points.")

print(f"Homography matrix calculated:")
print(H)
print(f"\nTransforming coordinates from video resolution ({args.video_width}x{args.video_height})")
print(f"to original image resolution ({args.img_width}x{args.img_height}), then to ground plane")

# Scale factors to convert from video resolution to original image resolution
scale_x = args.img_width / args.video_width
scale_y = args.img_height / args.video_height

# Read input CSV and transform coordinates
transformed_count = 0
with open(args.input, 'r') as infile, open(args.output, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Write header
    header = next(reader)
    writer.writerow(header)
    
    for row in reader:
        person_id = row[0]
        frame_idx = row[1]
        x_video = float(row[2])
        y_video = float(row[3])
        
        # Scale coordinates from video resolution to original image resolution
        # Both are in top-left origin coordinate system
        x_img = x_video * scale_x
        y_img = y_video * scale_y
        
        # Transform to homogeneous coordinates (already in top-left origin)
        point_img = np.array([[x_img, y_img]], dtype=np.float32)
        point_img = point_img.reshape(-1, 1, 2)
        
        # Apply homography transformation
        point_ground = cv2.perspectiveTransform(point_img, H)
        
        # Extract ground coordinates (now in top-left origin world coordinates)
        x_ground = point_ground[0][0][0]
        y_ground = point_ground[0][0][1]
        
        # Write transformed coordinates
        writer.writerow([person_id, frame_idx, x_ground, y_ground])
        transformed_count += 1
        
        if transformed_count % 10000 == 0:
            print(f"Transformed {transformed_count} coordinates...")

print(f"\nTransformation complete!")
print(f"Total coordinates transformed: {transformed_count}")
print(f"Output saved to: {args.output}")

