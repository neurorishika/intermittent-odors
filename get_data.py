from subprocess import call
from tqdm import tqdm
import os
import requests

filenames = ["data/data.zip.001","data/data.zip.002","data/data.zip.003",
             "data/data.zip.004","data/data.zip.005","data/data.zip.006",
             "data/data.zip.007","data/data.zip.008","data/data.zip.009",
             "data/data.zip.010","data/data.zip.011","data/data.zip.012"]

urls = ["https://osf.io/n35e4/download","https://osf.io/ku2rm/download","https://osf.io/w9p8k/download",
        "https://osf.io/6ngyq/download","https://osf.io/78edu/download","https://osf.io/7t9jh/download",
        "https://osf.io/4ukh5/download","https://osf.io/5z2e8/download","https://osf.io/3cn9w/download",
        "https://osf.io/yv3fn/download","https://osf.io/ez9c5/download","https://osf.io/k37yn/download"]

for fname, url in tqdm(zip(filenames, urls)):
  if not os.path.isfile(fname):
    try:
      r = requests.get(url)
    except requests.ConnectionError:
      print("!!! Failed to download data !!!")
    else:
      if r.status_code != requests.codes.ok:
        print("!!! Failed to download data !!!")
      else:
        with open(fname, "wb") as fid:
          fid.write(r.content)

dir_7zip  = "C:\\Program Files\\7-Zip\\"
os.chdir('data')
call(['set',f'PATH=%PATH%;{dir_7zip}'])
call(['7z','e','data.zip.001'])
for f in filter(lambda v: "data.zip." in v, os.listdir()):
  os.remove(f)