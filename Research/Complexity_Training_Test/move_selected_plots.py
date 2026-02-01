import os
import shutil

# Base directory where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Source directory where the plots are currently located
source_dir = os.path.join(base_dir, "plots")

# Destination directory for the selected plots
destination_folder = os.path.join(base_dir, "selected_plots")

# List of files to move
files_to_move = [
    "comparison_train_val_test_loss_faceted.png",
    "barplot_combined_losses.png",
    "comparison_complexity_metrics_faceted.png",
    "mechanism_adaptive_control_faceted.png"
]

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
    print(f"Created directory: {destination_folder}")

# Move the files
for filename in files_to_move:
    source_path = os.path.join(source_dir, filename)
    destination_path = os.path.join(destination_folder, filename)
    
    if os.path.exists(source_path):
        try:
            # Remove destination if it exists to avoid errors
            if os.path.exists(destination_path):
                os.remove(destination_path)
                
            shutil.move(source_path, destination_path)
            print(f"Moved: {filename}")
        except Exception as e:
            print(f"Error moving {filename}: {e}")
    else:
        print(f"File not found in source: {source_path}")
