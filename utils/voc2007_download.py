import os
import requests
import tarfile

def download_file(url, target_folder):
    local_filename = url.split('/')[-1]
    local_filepath = os.path.join(target_folder, local_filename)

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return local_filepath

def extract_tar_file(file_path, target_folder):
    with tarfile.open(file_path, 'r') as tar:
        tar.extractall(target_folder)

def download_voc2007(target_folder):
    base_url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/'
    files_to_download = ['VOCtrainval_06-Nov-2007.tar', 'VOCtest_06-Nov-2007.tar', 'VOCdevkit_08-Jun-2007.tar']

    for file in files_to_download:
        url = base_url + file
        print(f'Downloading {url}')
        file_path = download_file(url, target_folder)
        print(f'Extracting {file_path}')
        extract_tar_file(file_path, target_folder)
        print(f'Finished {file}')

if __name__ == "__main__":
    target_folder = 'VOC2007'
    download_voc2007(target_folder)
