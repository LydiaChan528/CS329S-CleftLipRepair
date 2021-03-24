# Federated Learning for Cleft Lip Repair

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Usage](#usage)
- [Team](#team)

## Overview

CS 329S Final Project Codebase

- Data Explorer (explorer.py)
    Allows browsing of dataset images, labels, and metadata

- model/
    Contains HRNet model trained to predict cleft lip landmarks

## External References

[elerac/fullscreen](https://github.com/elerac/fullscreen)
- Displaying images in fullscreen on projector
- Source contained in `calibration/fullscreen/`

[dclemmon/projection_mapping](https://github.com/dclemmon/projection_mapping)
- Initial implementation of camera calibration and perspective transformation
- Modified source files in `calibration/`


## Setup

This is intended to be ran on the cs329s-vm instance.
Ensure your networking rules allow you to connect to Streamlit.
Follow this [link](https://github.com/cs231n/gcloud/#configure-networking)
to setup networking rules.

You will find images and their annotations in
`/home/shared/cleft_images/`

You will find videos in
`/home/shared/cleft_videos/`

You will find associated metadata in
`/home/shared/metadata/`

You may need to change permissions on these files for your account.
Go to the top level `/home/shared/` and execute the following command:
`sudo chmod -R g+rw *`

You may want to use a virtual environment or conda to manage your dependecies.

### Step 1 - Install Requirements

`pip install -r requirements.txt`

### Step 2 - Install HRNet Requirements

NOTE: This step is not necessary if you are on cs329s-vm

`source model/setup.bashrc`

## Usage

### Data Explorer App

`streamlit run explorer.py`

Open the External IP Address displayed in your web browser


### Manual Model Inference on Validation Set

From `model/` run:

```
python tools/test.py --cfg experiments/cleft/face_alignment_cleft_hrnet_w18.yaml --model-file output/cleft/face_alignment_cleft_hrnet_w18/final_state.pth
```

You will find output images in
```
./model/output/cleft/face_alignment_cleft_hrnet_w18/output_images/projections
```



### Federated Learning

1. Create YAML for configuration (you can make a copy of experiment/fed_learn.yaml and modify as needed). Check that FEDERATED is True, CLIENTS is set to the number of clients to train with, and END_EPOCH is set as desired.

2. Train with federated learning
```
python tools/federated_train.py --cfg <config file from step 1>
```

3. Outputs will be saved to the according folder (./outputs/cleft/<config filename>). These include a 'npy' with the validation stats, a 'pth' of the best-performing server model so far, and graphs of nme/loss. 

Validation stats has rows of [epoch, local client 0 loss, local client 0 nme, ... server loss, server nme] for each epoch that was run.

4. Run Tensorboard following Training Pipeline instructions, or Streamlit to display validation stats.


### Training Pipeline

1. Create YAML for configuration (you can also modify an existing configuration/YAML)
```
python3 model/train_pipeline.py --epochs 2
```

2. Train the model
```
python tools/train.py --cfg experiments/cleft/train_nopretrained.yaml
```

3. View training in tensorboard
In another terminal, open the tensorboard logs
```
tensorboard --logdir <log dir that you want to view>
```
In a third terminal, run so that you can view the tensorboards locally
```
<gcloud command from the instance> -- -NfL 6006:localhost:600
```
Now you should be able to view the tensorboard at http://localhost:6006/

4. Compare resulting projections with another model:
Run manual model inference on the validation set, update configs in compare\_projections.py as necessary, and compare predictions in Streamlit!
```
streamlit run projections_comparator.py --server.port <a port between 7000~9000 that is not 8501>
```
5. Keep on training, and have fun!

## Team

- [@eldrickm](https://github.com/eldrickm)
- [@priscillalui](https://github.com/priscillalui)
- [@lydiachan528](https://github.com/lydiachan528)
- [@qjerry0](https://github.com/qjerry0)
