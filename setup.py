import os
import gdown
import zipfile

def download_from_drive(file_id, output_dir):
    url = f'https://drive.google.com/uc?id={file_id}'
    output_file = os.path.join(output_dir, f'{file_id}.zip')
    gdown.download(url, output_file, quiet=False)
    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    os.remove(output_file)

file_ids = ['1vxYmx9JAplUJ17cKxq3sqPExAh56Fj5q', '14-mdcNsYb8Kqk_E4aFFZDRExXj16qJMz']
folder_id = '1obzwntuvYR9zzZ4sBXeaCs5aekEzSlzb'
output_dir = os.getcwd()
for file_id in file_ids:
    download_from_drive(file_id, output_dir)
download_from_drive(folder_id, output_dir)
print("Files downloaded successfully!")