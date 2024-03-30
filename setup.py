import os
import gdown
import zipfile

file_id = '1vxYmx9JAplUJ17cKxq3sqPExAh56Fj5q'
drive_link = f'https://drive.google.com/uc?id={file_id}'
output_dir = "./"
os.makedirs(output_dir, exist_ok=True)
zip_file = os.path.join(output_dir, 'downloaded_file.zip') 
gdown.download(drive_link, zip_file, quiet=False)
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(output_dir)
os.remove(zip_file)
