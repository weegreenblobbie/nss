Make a python virtual env and install the requirements.

These examples are windows powershell format:

Align one tif against the target:

```
python align.py -t "C:\Users\weegr\Desktop\tmp_export\20250313\tiff-raw-totality-1\20250313_232842.99.tif" `
                   "C:\Users\weegr\Desktop\tmp_export\20250313\tiff-raw-totality-1\20250313_232746.99.tif"
```

Align multiple tifs agains the target:
```
python align.py -t TARGET_TIF TIFF1+
```

Align a bunch of files using python multiprocessing
```
python nss.py "C:\Users\weegr\Desktop\tmp_export\20250313\tiff-raw-totality-1\*20250313_232842.99*.tif"
```
