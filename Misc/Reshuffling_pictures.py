# import os
# import shutil

# # Base path where the Paintings directory is located
# base_path = '/Users/dhyanpatel/Desktop/IR/sanskriti/Image Data'

# # Get a list of painting types from the 'testing' directory, ignoring any files
# painting_types = next(os.walk(os.path.join(base_path, 'Paintings/testing')))[1]

# # Move the 'testing' and 'training' folders for each painting type into a new main subfolder
# for painting in painting_types:
#     # Create new main subfolder for the painting type if it doesn't exist
#     new_main_subfolder_path = os.path.join(base_path, 'Paintings', painting)
#     if not os.path.exists(new_main_subfolder_path):
#         os.makedirs(new_main_subfolder_path)
    
#     # Define the old testing and training paths
#     old_testing_path = os.path.join(base_path, 'Paintings/testing', painting)
#     old_training_path = os.path.join(base_path, 'Paintings/training', painting)
    
#     # Define the new testing and training paths
#     new_testing_path = os.path.join(new_main_subfolder_path, 'testing')
#     new_training_path = os.path.join(new_main_subfolder_path, 'training')
    
#     # Move the old testing and training folders to the new location, if they exist
#     if os.path.exists(old_testing_path):
#         shutil.move(old_testing_path, new_testing_path)
#     if os.path.exists(old_training_path):
#         shutil.move(old_training_path, new_training_path)



import os
import shutil

# Base path where the Paintings directory is located
base_path = '/Users/dhyanpatel/Desktop/IR/sanskriti/Image Data'

# Function to append a suffix before the file extension
def append_suffix(filename, suffix):
    name, ext = os.path.splitext(filename)
    return f"{name}{suffix}{ext}"

# Get a list of painting types from the 'testing' directory, ignoring any files
painting_types = next(os.walk(os.path.join(base_path, 'Monuments/test')))[1]

# Combine images from 'testing' and 'training' into a single folder for each painting type
for painting in painting_types:
    # Create new main subfolder for the painting type if it doesn't exist
    painting_folder_path = os.path.join(base_path, 'Monuments', painting)
    os.makedirs(painting_folder_path, exist_ok=True)
    
    # Define the old testing and training paths
    old_testing_path = os.path.join(base_path, 'Monuments/test', painting)
    old_training_path = os.path.join(base_path, 'Monuments/train', painting)
    
    # Move and rename files from the old testing folder
    for filename in os.listdir(old_testing_path):
        src_file = os.path.join(old_testing_path, filename)
        new_filename = append_suffix(filename, 'TS') # Append 'TS' for testing
        dest_file = os.path.join(painting_folder_path, new_filename)
        shutil.move(src_file, dest_file)
    
    # Move and rename files from the old training folder
    for filename in os.listdir(old_training_path):
        src_file = os.path.join(old_training_path, filename)
        new_filename = append_suffix(filename, 'TN') # Append 'TN' for training
        dest_file = os.path.join(painting_folder_path, new_filename)
        shutil.move(src_file, dest_file)

# After processing, the 'testing' and 'training' folders will be empty and can be removed if desired
# shutil.rmtree(os.path.join(base_path, 'Paintings/testing'))
# shutil.rmtree(os.path.join(base_path, 'Paintings/training'))
