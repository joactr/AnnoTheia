<h1 align="left"> 📜 Fine-Tuning TalkNet-ASD to the Language I am Interested On</h1>

In this tutorial, we are goint to learn how to fine-tune the [TalkNet-ASD model](https://dl.acm.org/doi/abs/10.1145/3474085.3475587) to our language of interest. To do this, here you can find different tips regarding how we can collect data, as well as the code for data preparation, model training and its incorporation into the toolkit.

## 🦺 Conda Environment Setup

Just in case we are not where we should:

```
cd ./modules/active_speaker_detection/talknet_asd/
conda activate annotheia
```

## ⛏️ Data Collection

As you could expect, the first thing we need is data. This is the most complicated of the steps in this tutorial. Find the following tips here:

- **What do we need?** → Videos where only one person appears on scene and is speaking
- **Where can we find these types of videos?** → One way is to search social media for the most popular vloggers who upload videos in the language of your interest. Please ask for permission if you plan to publish this data.
- **How many data do we need?** → Of course, the more, the better. However, **please note that this ASD models work at window level** and usually these windows do not span more than 2 seconds. So the good news is that from a small number of vlogs we will be able to extract hundreds of window samples to estimate our TalkNet-ASD model.
- **Anything else?** → Try to collect as many different speakers as possible to estimate a model robust against people it has never seen. For this reason, you should **keep in mind this speaker independency** when creating the dataset splits.

Once you have collected your data (Congratulations👏!), in order to run the scripts described in the following steps of this tutorial, please **organize your videos like** this structure scheme:

```
videos_for_spanish/
├── training/
│   ├── speaker000/
│   │   ├── speaker000_0000.mp4
│   │   ├── speaker000_0001.mp4
│   │   ├── ...
│   ├── speaker001/
│   │   ├── ...
│   ├── ...
├── validation/
│   ├── speaker020/
│   │   ├── speaker020_0000.mp4
│   │   ├── ...
│   ├── ...
├── test/
│   ├── speaker025/
│   │   ├── speaker030_0000.mp4
│   │   ├── ...
│   ├── ...
```
🌟 **Tip:** If your videos are very long, please split each one into chunks of 20 seconds using the ```ffmpeg``` package.

## 🛠️ Data Preparation

In this section, we are going to prepare all the things we need to fine-tune our pre-trained TalkNet-ASD model:

- **Extracting Face Crops.** Running the following script we will obtain the 112x112 face crops TalkNet-ASD is expecting:

```
python ./scripts/extract_face_crops.py --video-dir ${PATH_TO_VIDEO_DIR}
```

- **Extracting MFCCs.** Running the following script we will extract the 13 Mel Frequency Cepstral Coefficients (MFCCs) at 100fps TalkNet-ASD is expecting:

```
python ./scripts/extract_mfccs.py --video-dir ${PATH_TO_VIDEO_DIR}
```
🌟 **Detail:** Both the face crops and MFCCs are stored following the directory structure described above but data is saved as compressed ```.npz``` files.


## 📝 Citation
If you find interesting this tutorial, **please cite the original work** where TalkNet-ASD was presented:

```
@inproceedings{tao2021someone,
  title={Is Someone Speaking? Exploring Long-term Temporal Features for Audio-visual Active Speaker Detection},
  author={Tao, Ruijie and Pan, Zexu and Das, Rohan Kumar and Qian, Xinyuan and Shou, Mike Zheng and Li, Haizhou},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={3927–3935},
  year={2021},
}
```


