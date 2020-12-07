# Image recognition task for COMP.SGN.240

This repository contains my code for an assignment for Advanced signal processing course at Tampere University. The assignment involves building an image recognition method to categorize images as "smile" or "non-smile". The assignment also require a live demo using a web cam or a video file. This repository implements a web cam demo.

The instruction can be viewed [here](https://moodle.tuni.fi/pluginfile.php/841987/mod_resource/content/5/COMP.SGN.240_Image_Recognition_2020.pdf) (requires Moodle sign-in).

## Getting started

Use `download_genki4K.sh` in bash to download and extract the GENKI-4K dataset into the folder `genki4k/`. You can also download the dataset [manually](https://inc.ucsd.edu/mplab/398.php).

To install requirements, run `pip install -r requirements.txt`.

To train a model, run `python main.py`. After training a model, you can try the live demo by running `python live_dmeo.py` (requires web cam).

## About the configuration

The configuration can be changed in `config/main_settings.yaml`. Some examples:

 - To change the name of the saved model, change the `method_name` field.
 - To use/not use a pretrained extractor, change the `network/pretrained_name` field
 - To use cpu/gpu, change the field `training/device` to the appropriate value (cpu/gpu)
