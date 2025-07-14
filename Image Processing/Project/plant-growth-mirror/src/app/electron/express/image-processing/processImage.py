#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image processing script for plant growth analysis.
Compatible with Python 3.8+
"""

import sys
import os
import json
import base64

def check_python_version():
    """Check if the Python version is compatible (>=3.8). Exit with error if not."""
    if sys.version_info < (3, 8):
        print(json.dumps({"error": f"Python 3.8+ required, but {sys.version_info.major}.{sys.version_info.minor} found"}), file=sys.stderr)
        sys.exit(1)

def import_dependencies():
    """Try to import required dependencies, exit with error if not installed."""
    try:
        global cv2, np
        import cv2
        import numpy as np
    except ImportError as e:
        print(json.dumps({"error": f"Missing required package: {str(e)}. Please ensure opencv-python and numpy are installed."}), file=sys.stderr)
        sys.exit(1)

def error_exit(message):
    """Exit with error message in JSON format."""
    error_output = {
        "dataProcessedBase64": "",
        "result": {
            "height": 0,
            "width": 0,
            "area": 0,
        },
        "error": message
    }
    print(json.dumps(error_output), file=sys.stderr)
    sys.exit(1)

def parse_arguments():
    """Parse and validate command-line arguments. Returns the image_id."""
    if len(sys.argv) != 2:
        error_exit("Missing image ID")
    return sys.argv[1]

def load_json_input(image_id, script_dir):
    """Load and parse the input JSON file for the given image_id."""
    json_path = os.path.join(script_dir, "temp", f"{image_id}.json")
    if not os.path.exists(json_path):
        error_exit(f"Input JSON not found: {json_path}")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        error_exit(f"Error reading JSON: {str(e)}")
    return data

def decode_base64_image(base64_image):
    """Decode a base64-encoded image string to a NumPy array (OpenCV image)."""
    # Strip base64 header if present
    if base64_image.startswith("data:"):
        parts = base64_image.split(",", 1)
        if len(parts) == 2:
            base64_image = parts[1]
    try:
        img_bytes = base64.b64decode(base64_image)
        if len(img_bytes) == 0:
            error_exit("Decoded base64 image is empty")
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        if len(arr) == 0:
            error_exit("Image buffer is empty after decoding")
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        error_exit(f"Error decoding base64 image: {str(e)}")
    if img is None:
        error_exit("Decoded image is None (cv2.imdecode failed)")
    
    # Adaptive resize for efficiency - limit to 1024px on largest dimension
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim > 1024:
        scale = 1024.0 / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return img

def validate_image(img):
    """Validate that the image is a 3-channel color image."""
    if len(img.shape) != 3 or img.shape[2] != 3:
        error_exit(f"Invalid image shape: {img.shape}. Expected 3-channel color image.")

def preprocess_image(img):
    """Convert image to grayscale and apply enhanced preprocessing."""
    # Convert to grayscale with weighted average optimized for vegetation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive Gaussian blur based on image size
    h, w = img.shape[:2]
    kernel_size = max(3, min(7, (h + w) // 400))
    if kernel_size % 2 == 0:
        kernel_size += 1
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    
    # Apply CLAHE for better contrast in plant regions
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    
    return enhanced, gray

def edge_detection(blur):
    """Apply Sobel filter for edge detection"""
    grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    grad = cv2.convertScaleAbs(grad)
    
    # Optimized morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad_closed = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return grad_closed

def create_markers(gray, granularity):
    """Create marker image for watershed segmentation based on granularity."""
    h, w = gray.shape
    markers = np.zeros((h, w), dtype=np.int32)
    
    # Adaptive step size based on image dimensions and granularity
    step = max(int(granularity * min(h, w) / 500), 4)
    radius = max(step // 32, 1)
    
    # Create markers with better distribution
    label = 2
    y_coords = np.arange(step // 2, h, step)
    x_coords = np.arange(step // 2, w, step)
    
    # Vectorized marker creation for better performance
    for y in y_coords:
        for x in x_coords:
            cv2.circle(markers, (x, y), radius, int(label), -1)
            label += 1
    
    # Add border markers for better segmentation
    markers[0, :] = 1
    markers[-1, :] = 1
    markers[:, 0] = 1
    markers[:, -1] = 1
    
    return markers

def watershed_segmentation(grad_closed, markers):
    """Apply watershed segmentation using the provided markers."""
    try:
        markers = cv2.watershed(cv2.cvtColor(grad_closed, cv2.COLOR_GRAY2BGR), markers)
    except Exception as e:
        error_exit(f"Watershed segmentation failed: {str(e)}")
    return markers

def compute_region_colors(img, markers, threshold):
    """Compute mean color for each region with enhanced green detection."""
    unique_labels = np.unique(markers)
    unique_labels = unique_labels[unique_labels > 1]
    if len(unique_labels) == 0:
        error_exit("No valid regions found in watershed segmentation")
    
    # Convert to HSV for better green detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    flat_markers = markers.flatten()
    flat_bgr = img.reshape(-1, 3)
    flat_hsv = hsv.reshape(-1, 3)
    valid_mask = flat_markers > 1
    labels_for_count = flat_markers[valid_mask]
    
    if len(labels_for_count) == 0:
        error_exit("No valid labels found")
    
    max_label = labels_for_count.max()
    
    # Compute statistics more efficiently
    counts = np.bincount(labels_for_count)
    sum_b = np.bincount(labels_for_count, weights=flat_bgr[valid_mask, 0])
    sum_g = np.bincount(labels_for_count, weights=flat_bgr[valid_mask, 1])
    sum_r = np.bincount(labels_for_count, weights=flat_bgr[valid_mask, 2])
    sum_h = np.bincount(labels_for_count, weights=flat_hsv[valid_mask, 0])
    sum_s = np.bincount(labels_for_count, weights=flat_hsv[valid_mask, 1])
    sum_v = np.bincount(labels_for_count, weights=flat_hsv[valid_mask, 2])
    
    nonzero_mask = counts > 0
    
    # Vectorized mean calculations
    mean_b = np.divide(sum_b, counts, out=np.zeros_like(sum_b, dtype=np.float64), where=nonzero_mask)
    mean_g = np.divide(sum_g, counts, out=np.zeros_like(sum_g, dtype=np.float64), where=nonzero_mask)
    mean_r = np.divide(sum_r, counts, out=np.zeros_like(sum_r, dtype=np.float64), where=nonzero_mask)
    mean_h = np.divide(sum_h, counts, out=np.zeros_like(sum_h, dtype=np.float64), where=nonzero_mask)
    mean_s = np.divide(sum_s, counts, out=np.zeros_like(sum_s, dtype=np.float64), where=nonzero_mask)
    mean_v = np.divide(sum_v, counts, out=np.zeros_like(sum_v, dtype=np.float64), where=nonzero_mask)
    
    colors = np.zeros((max_label + 1, 3), dtype=np.uint8)
    
    # Enhanced green detection using multiple criteria
    # BGR criteria (original)
    bgr_green = (mean_g > mean_b * (1 + threshold)) & (mean_g > mean_r * (1 + threshold))
    
    # HSV criteria - green hue range and sufficient saturation
    hsv_green = ((mean_h >= 40) & (mean_h <= 80)) & (mean_s >= 30) & (mean_v >= 20)
    
    # Combined criteria with adaptive threshold
    green_condition = bgr_green | hsv_green
    
    # Apply more lenient threshold for regions with high saturation
    high_sat_mask = mean_s >= 60
    lenient_bgr = (mean_g > mean_b * (1 + threshold * 0.7)) & (mean_g > mean_r * (1 + threshold * 0.7))
    green_condition = green_condition | (high_sat_mask & lenient_bgr)
    
    for lbl in unique_labels:
        if lbl <= max_label and counts[lbl] > 0:
            if green_condition[lbl]:
                colors[lbl] = [mean_b[lbl], mean_g[lbl], mean_r[lbl]]
            else:
                colors[lbl] = [0, 0, 0]
    
    seg = colors[markers]
    return seg

def fill_green_holes(seg):
    """Fill holes in the green mask with optimized morphological operations."""
    # Create green mask more efficiently
    green_mask = (seg[:, :, 1] > seg[:, :, 0]) & (seg[:, :, 1] > seg[:, :, 2])
    green_mask = green_mask.astype(np.uint8) * 255
    
    # Use more efficient morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Fill small holes using morphological closing
    closed_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find and fill larger holes
    inv_mask = cv2.bitwise_not(closed_mask)
    contours, hierarchy = cv2.findContours(inv_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is not None:
        # Adaptive minimum area based on image size
        h, w = seg.shape[:2]
        min_area = max(50, (h * w) // 5000)
        
        for idx, hr in enumerate(hierarchy[0]):
            if hr[3] != -1:  # Inner contour (hole)
                cnt = contours[idx]
                if cv2.contourArea(cnt) >= min_area:
                    cv2.drawContours(closed_mask, contours, idx, 255, -1)
    
    # Apply the filled mask to the segmentation
    hole_mask = (closed_mask == 255) & np.all(seg == 0, axis=2)
    seg[hole_mask] = [0, 255, 0]
    
    return seg

def overlay_segmentation(img, seg):
    """Overlay the segmentation result on the original image."""
    return cv2.addWeighted(img, 0.3, seg, 0.7, 0)

def extract_plant_pixels(seg):
    """Extract plant pixel coordinates from segmentation."""
    mask_nonzero = np.any(seg != 0, axis=2)
    ys, xs = np.where(mask_nonzero)
    return xs, ys

def perform_linear_regression(xs, ys):
    """Perform linear regression on plant pixels to find main axis."""
    try:
        a, b = np.polyfit(xs, ys, deg=1)
        return a, b, True
    except np.linalg.LinAlgError:
        return None, None, False

def draw_fallback_measurement_lines(overlay, xs, ys):
    """Draw simple bounding box measurement lines when regression fails."""
    top_point = (xs[np.argmin(ys)], ys.min())
    bottom_point = (xs[np.argmax(ys)], ys.max())
    left_point = (xs.min(), ys[np.argmin(xs)])
    right_point = (xs.max(), ys[np.argmax(xs)])
    
    cv2.line(overlay, bottom_point, top_point, (0, 255, 255), 2)  # Yellow line
    cv2.line(overlay, left_point, right_point, (255, 0, 255), 2)  # Magenta line
    return overlay

def calculate_height_line_points(xs, ys, a, b):
    """Calculate height line endpoints using linear regression projection."""
    # Define reference point and unit vector
    p0 = np.array([0, b])  # Reference point (x=0, y=b)
    v = np.array([1, a])   # Direction vector of the line y = a*x + b
    u = v / np.hypot(*v)   # Unit vector
    
    # Create array of segmented points
    pts = np.vstack((xs, ys)).T  # shape (N,2)
    
    # Project all points onto the unit vector: t_i = (pt_i - p0) · u
    t = (pts - p0).dot(u)
    
    # Find extreme projections for height
    idx_min, idx_max = np.argmin(t), np.argmax(t)
    t_min, t_max = t[idx_min], t[idx_max]
    
    # Compute the points on the line corresponding to these projections
    pt_start = p0 + u * t_min
    pt_end = p0 + u * t_max
    pt_start = (int(pt_start[0]), int(pt_start[1]))
    pt_end = (int(pt_end[0]), int(pt_end[1]))
    
    return pt_start, pt_end

def calculate_width_line_points(xs, ys, a, b):
    """Calculate width line endpoints using perpendicular projection."""
    # Define reference point and unit vectors
    p0 = np.array([0, b])  # Reference point (x=0, y=b)
    v = np.array([1, a])   # Direction vector of the line y = a*x + b
    u = v / np.hypot(*v)   # Unit vector
    
    # Unit vector perpendicular to the main axis (rotate u by 90°)
    u_perp = np.array([-u[1], u[0]])
    
    # Create array of segmented points
    pts = np.vstack((xs, ys)).T  # shape (N,2)
    
    # Project each point onto the perpendicular direction: s_i = (pt_i - p0) · u_perp
    s = (pts - p0).dot(u_perp)
    
    # Find extreme projections for width
    idx_min_perp = np.argmin(s)
    idx_max_perp = np.argmax(s)
    
    # Get the actual pixel coordinates for width extremes
    pt_left = (int(pts[idx_min_perp][0]), int(pts[idx_min_perp][1]))
    pt_right = (int(pts[idx_max_perp][0]), int(pts[idx_max_perp][1]))
    
    return pt_left, pt_right

def draw_line_with_endpoints(overlay, pt_start, pt_end, color, thickness=2, circle_radius=4):
    """Draw a line with circular endpoints on the overlay."""
    cv2.line(overlay, pt_start, pt_end, color, thickness)
    cv2.circle(overlay, pt_start, circle_radius, color, -1)
    cv2.circle(overlay, pt_end, circle_radius, color, -1)

def draw_measurement_lines(overlay, seg):
    """Draw height and width measurement lines on the overlay image using linear regression."""
    # Extract plant pixel coordinates
    xs, ys = extract_plant_pixels(seg)
    
    if len(xs) == 0:
        return overlay  # No plant pixels found, return unchanged
    
    # Attempt linear regression to find the main axis of the plant
    a, b, regression_success = perform_linear_regression(xs, ys)
    
    if not regression_success:
        # Fallback to simple bounding box if regression fails
        return draw_fallback_measurement_lines(overlay, xs, ys)
    
    # Calculate height line points (along main axis)
    height_start, height_end = calculate_height_line_points(xs, ys, a, b)
    
    # Calculate width line points (perpendicular to main axis)
    width_left, width_right = calculate_width_line_points(xs, ys, a, b)
    
    # Draw height line (yellow)
    draw_line_with_endpoints(overlay, height_start, height_end, (0, 255, 255))
    
    # Draw width line (magenta)
    draw_line_with_endpoints(overlay, width_left, width_right, (255, 0, 255))
    
    return overlay

def extract_features(seg, img):
    """Extract height, width, and area with improved robustness."""
    mask_nonzero = np.any(seg != 0, axis=2)
    ys, xs = np.where(mask_nonzero)
    if len(xs) == 0:
        return 0.0, 0.0, 0
    
    # Use more robust fitting for height calculation
    if len(xs) < 3:
        # For very small plants, use simple bounding box
        altura = float(ys.max() - ys.min())
        largura = float(xs.max() - xs.min())
    else:
        # Use RANSAC for more robust line fitting in case of outliers
        try:
            # Downsample points for efficiency if too many
            if len(xs) > 1000:
                indices = np.random.choice(len(xs), 1000, replace=False)
                xs_sample = xs[indices]
                ys_sample = ys[indices]
            else:
                xs_sample = xs
                ys_sample = ys
            
            # Robust polynomial fitting
            a, b = np.polyfit(xs_sample, ys_sample, 1)
            v = np.array([1, a])
            u = v / np.hypot(*v)
            pts = np.vstack((xs, ys)).T
            p0 = np.array([0, b])
            t = (pts - p0).dot(u)
            altura = float(t.max() - t.min())
            
            # Calculate width using perpendicular direction
            v_perp = np.array([-u[1], u[0]])
            s = (pts - p0).dot(v_perp)
            idx_min, idx_max = int(np.argmin(s)), int(np.argmax(s))
            pt_left = pts[idx_min]
            pt_right = pts[idx_max]
            largura = float(np.linalg.norm(pt_right - pt_left))
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to bounding box if fitting fails
            altura = float(ys.max() - ys.min())
            largura = float(xs.max() - xs.min())
    
    area_pixels = int(np.count_nonzero(mask_nonzero))
    return altura, largura, area_pixels

def encode_image_to_base64(overlay):
    """Encode the overlay image to base64 with optimized compression."""
    try:
        # Use JPEG compression for smaller file size while maintaining quality
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        success, buffer = cv2.imencode('.jpg', overlay, encode_params)
        if not success or buffer is None:
            # Fallback to PNG if JPEG fails
            success, buffer = cv2.imencode('.png', overlay)
            if not success or buffer is None:
                error_exit("Failed to encode overlay image")
            overlay_base64 = "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode('utf-8')
        else:
            overlay_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode('utf-8')
    except Exception as e:
        error_exit(f"Error encoding overlay image: {str(e)}")
    return overlay_base64

def validate_and_format_results(altura, largura, area_pixels):
    """Validate and format the extracted features for output."""
    import numpy as np
    try:
        height_val = float(altura) if np.isfinite(altura) else 0.0
        width_val = float(largura) if np.isfinite(largura) else 0.0
        area_val = int(area_pixels) if np.isfinite(area_pixels) else 0
    except (ValueError, OverflowError):
        height_val = 0.0
        width_val = 0.0
        area_val = 0
    return height_val, width_val, area_val

def load_and_validate_input():
    """Orchestrate argument parsing, JSON loading, and input validation."""
    image_id = parse_arguments()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data = load_json_input(image_id, script_dir)
    base64_image = data.get("dataBase64")
    granularity = int(data.get("granularity", 20))
    threshold = float(data.get("threshold", 0.05))
    if base64_image is None:
        error_exit("Missing 'dataBase64' in JSON")
    return base64_image, granularity, threshold

def process_image_pipeline(base64_image, granularity, threshold):
    """Orchestrate the image decoding, preprocessing, segmentation, and feature extraction."""
    img = decode_base64_image(base64_image)
    validate_image(img)
    blur, gray = preprocess_image(img)
    grad_closed = edge_detection(blur)
    markers = create_markers(gray, granularity)
    markers = watershed_segmentation(grad_closed, markers)
    seg = compute_region_colors(img, markers, threshold)
    seg = fill_green_holes(seg)
    overlay = overlay_segmentation(img, seg)
    overlay = draw_measurement_lines(overlay, seg)  # Add measurement lines
    altura, largura, area_pixels = extract_features(seg, img)
    overlay_base64 = encode_image_to_base64(overlay)
    height_val, width_val, area_val = validate_and_format_results(altura, largura, area_pixels)
    return overlay_base64, height_val, width_val, area_val

def output_results(overlay_base64, height_val, width_val, area_val):
    """Format and print the final output as JSON."""
    output = {
        "dataProcessedBase64": overlay_base64,
        "result": {
            "height": round(height_val, 2),
            "width": round(width_val, 2),
            "area": area_val
        }
    }
    try:
        print(json.dumps(output, ensure_ascii=False, separators=(',', ':')))
    except Exception as e:
        error_exit(f"Error serializing output JSON: {str(e)}")
    sys.exit(0)

def main():
    """Main function to orchestrate the image processing pipeline."""
    check_python_version()
    import_dependencies()
    base64_image, granularity, threshold = load_and_validate_input()
    overlay_base64, height_val, width_val, area_val = process_image_pipeline(base64_image, granularity, threshold)
    output_results(overlay_base64, height_val, width_val, area_val)

if __name__ == "__main__":
    main()
