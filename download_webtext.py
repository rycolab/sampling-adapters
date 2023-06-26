import os
import requests
from tqdm import tqdm

DATA_URL = "https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/"
subdir = 'data'

if __name__ == '__main__':
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir = subdir.replace('\\','/') # needed for Windows

    for ds in ['webtext']:
        for split in ['test']:
            filename = ds + "." + split + '.jsonl'
            r = requests.get(DATA_URL + filename, stream=True)

            with open(os.path.join(subdir, filename), 'wb') as f:
                file_size = int(r.headers["content-length"])
                chunk_size = 1000
                with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                    # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(chunk_size)