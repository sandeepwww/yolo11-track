import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from collections import Counter

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot x/y heatmap of ground plane coordinates')
parser.add_argument('--input', type=str, default='tracks_ground.csv', help='Input CSV file with ground coordinates')
parser.add_argument('--output', type=str, default='ground_distribution.png', help='Output plot file')
parser.add_argument('--dpi', type=int, default=150, help='Output image DPI (default: 150)')
parser.add_argument('--bins', type=int, default=100, help='Number of bins for heatmap (default: 100)')
parser.add_argument('--outlier-percentile', type=float, default=1.0, help='Percentile to remove outliers (default: 1.0)')
args = parser.parse_args()

# Read ground coordinates from CSV and count occurrences
print(f"Reading coordinates from {args.input}...")
coordinate_counts = Counter()

with open(args.input, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    
    for row in reader:
        x = float(row[2])
        y = float(row[3])
        # Round to reasonable precision for grouping (0.1 meter precision)
        x_rounded = round(x, 1)
        y_rounded = round(y, 1)
        coordinate_counts[(x_rounded, y_rounded)] += 1

print(f"Loaded {sum(coordinate_counts.values())} total coordinate entries")
print(f"Unique coordinate pairs: {len(coordinate_counts)}")

# Extract coordinates and counts
x_coords = np.array([coord[0] for coord in coordinate_counts.keys()])
y_coords = np.array([coord[1] for coord in coordinate_counts.keys()])
counts = np.array(list(coordinate_counts.values()))

# Filter outliers based on percentiles
x_lower = np.percentile(x_coords, args.outlier_percentile)
x_upper = np.percentile(x_coords, 100 - args.outlier_percentile)
y_lower = np.percentile(y_coords, args.outlier_percentile)
y_upper = np.percentile(y_coords, 100 - args.outlier_percentile)

mask = (x_coords >= x_lower) & (x_coords <= x_upper) & (y_coords >= y_lower) & (y_coords <= y_upper)
x_filtered = x_coords[mask]
y_filtered = y_coords[mask]
counts_filtered = counts[mask]

print(f"After filtering outliers ({args.outlier_percentile}%): {len(x_filtered)} coordinate pairs")
print(f"X range: [{np.min(x_filtered):.2f}, {np.max(x_filtered):.2f}]")
print(f"Y range: [{np.min(y_filtered):.2f}, {np.max(y_filtered):.2f}]")
print(f"Count range: [{np.min(counts_filtered)}, {np.max(counts_filtered)}]")

# Create 2D histogram weighted by counts
hist, xedges, yedges = np.histogram2d(
    x_filtered, y_filtered, 
    bins=args.bins,
    weights=counts_filtered
)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(hist.T, origin='lower', extent=extent, aspect='auto', cmap='hot', interpolation='bilinear')
ax.set_xlabel('X (meters)', fontsize=12)
ax.set_ylabel('Y (meters)', fontsize=12)
ax.set_title('Ground Plane Presence Heatmap (Count of Occurrences)', fontsize=14)
cbar = plt.colorbar(im, ax=ax, label='Occurrence Count')
cbar.formatter.set_powerlimits((0, 0))

# Save the plot
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

plt.show()

