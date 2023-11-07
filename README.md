# Token-Mixer: Bind Image and Text in One Embedding Space for Medical Image Reporting (submitted to TMI-2023)
The Pytorch Implementation of Token-Mixer. 

## Introduction
In this project, we use Ubuntu 16.04.5, Python 3.7, Pytorch 1.8.1 and four NVIDIA RTX 2080Ti GPU. 

## Datasets
The medical image report generation datasets are available at the following links:
1. MIMIC-CXR-JPG data can be found at https://physionet.org/content/mimic-cxr-jpg/2.0.0/.
2. IU X-Ray data can be found at https://openi.nlm.nih.gov/.
3. Bladder Pathology data can be found at https://figshare.com/projects/nmi-wsi-diagnosis/61973.

### Training

To train the model, you need to prepare our training dataset.

Check the dataset path in train.py, and then run:
```
python train.py
```

### Testing

Check the model and data path in test.py, and then run:

```
python test.py
```

# Dependencies
  - Python=3.7.3
  - pytorch=1.8.1
  - pickle
  - tqdm
  - time
  - argparse
  - sklearn
  - json
  - numpy 
  - torchvision 
  - itertools
  - collections
  - math
  - os
  - matplotlib
  - PIL 
  - itertools
  - copy
  - re
  - abc
  - pandas
  - torch

# the metric meteor
the paraphrase-en.gz should be put into the .\pycocoevalcap\meteor\data, since the file is too big to upload.
