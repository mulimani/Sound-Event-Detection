# **Sound Event Detection using Convolutional Recurrent Neural Networks**

This repo contains the simple code for [Sound Event Detection](https://dcase.community/challenge2017/task-sound-event-detection-in-real-life-audio) written using PyTorch. Performance of the model is evaluated using [sed_eval](https://tut-arg.github.io/sed_eval/) and [dcase_util](https://github.com/DCASE-REPO/dcase_util) toolboxes.

### Prerequisite

* torch>=1.11.0
* numpy>=1.19.5
* pandas>=1.3.4
* torchaudio>=0.11.0
* scikit-learn>=0.24.2
* sed_eval>=0.1.8
* dcase_util>=0.2.16
### Getting started

1. Install the requirements: pip install -r requirements.txt
2. Download the DCASE 2017 Task3 [development](https://zenodo.org/records/814831) and [evaluation](https://zenodo.org/records/1040179) datasets into  dataset/SED_2017_street folder.
3. Run experiment: python main.py --epoch 120 --batch-size 64 --num-workers 4

### Acknowledgement 

This code is built on [dcase_util](https://github.com/DCASE-REPO/dcase_util) toolbox