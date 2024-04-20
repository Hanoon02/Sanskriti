import os
import gdown
import zipfile

# Function to download files from Google Drive
def download_from_drive(file_id, output_dir):
    url = f'https://drive.google.com/uc?id={file_id}'
    output_file = os.path.join(output_dir, f'{file_id}.zip')
    gdown.download(url, output_file, quiet=False)
    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    os.remove(output_file)

# IDs of the files and folder to be downloaded
file_ids = ['1vxYmx9JAplUJ17cKxq3sqPExAh56Fj5q', '1Xrkv6nYSJBAZsgoEwJ44amnf5u9aSG-H']
folder_id = '1obzwntuvYR9zzZ4sBXeaCs5aekEzSlzb'

# Output directory (current directory)
output_dir = os.getcwd()

# Download files
for file_id in file_ids:
    download_from_drive(file_id, output_dir)

# Download folder
download_from_drive(folder_id, output_dir)

print("Files downloaded successfully!")
