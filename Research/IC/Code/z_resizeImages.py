import pyvips
import os
from pathlib import Path

def resize_images(folder_path, target_width=800, target_height=None, recursive=True):
    # Supported image formats
    supported_formats = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp', '.gif'}
    
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    # Get all image files in the folder (and subfolders if recursive)
    if recursive:
        image_files = [f for f in folder.rglob('*') 
                       if f.is_file() and f.suffix.lower() in supported_formats 
                       and '_resize' not in f.stem]
    else:
        image_files = [f for f in folder.iterdir() 
                       if f.is_file() and f.suffix.lower() in supported_formats 
                       and '_resize' not in f.stem]
    
    if not image_files:
        print(f"No images found in '{folder_path}'" + (" (including subfolders)" if recursive else ""))
        return
    
    print(f"Found {len(image_files)} images to resize.")
    
    resized_count = 0
    skipped_count = 0
    error_count = 0
    
    for img_path in image_files:
        try:
            # Load image
            image = pyvips.Image.new_from_file(str(img_path), access='sequential')
            
            # Skip if image is already smaller than target width
            if image.width <= target_width:
                print(f"⊘ Skipped: {img_path.relative_to(folder)} (already {image.width}x{image.height}, smaller than target)")
                skipped_count += 1
                continue
            
            # Calculate scale factor
            if target_height is None:
                # Maintain aspect ratio based on width
                scale = target_width / image.width
            else:
                # Use both width and height, maintaining aspect ratio
                scale_w = target_width / image.width
                scale_h = target_height / image.height
                scale = min(scale_w, scale_h)
            
            # Resize image using pyvips (very efficient for large images)
            resized = image.resize(scale)
            
            # Create output filename
            output_name = f"{img_path.stem}_resize{img_path.suffix}"
            output_path = img_path.parent / output_name
            
            # Save resized image
            resized.write_to_file(str(output_path))
            
            print(f"✓ Resized: {img_path.relative_to(folder)} -> {output_name} ({resized.width}x{resized.height})")
            resized_count += 1
            
        except Exception as e:
            print(f"✗ Error processing {img_path.relative_to(folder)}: {str(e)}")
            error_count += 1
    
    print(f"\n{'='*60}")
    print(f"Resize operation completed!")
    print(f"  ✓ Resized: {resized_count}")
    print(f"  ⊘ Skipped: {skipped_count}")
    print(f"  ✗ Errors:  {error_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Configuration
    SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
    INPUT_FOLDER = os.path.join(SCRIPT_FOLDER, 'stats_results')

    TARGET_WIDTH = 262144  # Change this to your desired width (e.g., 1920 for Full HD, 3840 for 4K)
    TARGET_HEIGHT = None  # Set to None to maintain aspect ratio, or specify a height
    
    # Run the resize operation
    resize_images(INPUT_FOLDER, TARGET_WIDTH, TARGET_HEIGHT)
