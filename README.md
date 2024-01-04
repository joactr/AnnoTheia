<h1 align="center"><span style="font-weight:normal">AnnoTheia</h1>
<h2 align="center">A Semi-Automatic Annotation Toolkit for Audio-Visual Speech Technologies</h2>    
<div align="center">
  
[José-Miguel Acosta-Triana](), [David Gimeno-Gómez](https://scholar.google.es/citations?user=DVRSla8AAAAJ&hl=en), [Carlos-D. Martínez-Hinarejos](https://scholar.google.es/citations?user=M_EmUoIAAAAJ&hl=en)
</div>

<div align="center">
  
[📘 Introduction](#intro) |
[🛠️ Preparation](#preparation) |
[🚀 Get on with it!](#getonwithit) |
[💕 How can I help?](#helping) |
[📖 Citation](#citation) |
[📝 License](#license)
</div>

## <a name="intro"></a> 📘 Introduction

## <a name="preparation"></a> 🛠️ Preparation

- Create and activate a new conda environment:

```
conda create -y -n annotheia python=3.10
conda activate annotheia
```
- Install all requirements to prepare the environment:

```
bash ./prepare_environment.bash
```

## <a name="getonwithit"></a> 🚀 Get on with it!

The AnnoTheia toolkit is divided into two stages:

- **Detect candidate scenes** to compile the new audio-visual database from long videos:

```
python main_scenes.py \
    --video_dir ${PATH_TO_VIDEO_DIR} \
    --config-file ${PATH_TO_CONFIG_FILE} \
    --output-dir ${PATH_TO_OUTPUT_DIR}
```

- **Supervise & Annotate** the candidate scenes detected by the toolkit. Once the the previous script warn you about a completed long video:

```
python main_gui.py --scenes-info-path ${PATH_TO_SCENES_INFO_CSV}
```
🌟 We plan to unify both stages. Any comments or suggestions in this regard will be of great help!

## <a name="helping"></a> 💕 How can I help?

### How many languages are we currently covering?

## <a name="citation"></a> 📖 Citation
If you found our work useful, please cite our paper:

[AnnoTheia: A Semi-Automatic Annotation Toolkit for Audio-Visual Speech Technologies]()

```
@inproceedings{acosta24annotheia,
  author="Acosta-Triana, José-Miguel and Gimeno-Gómez, David and Martínez-Hinarejos, Carlos-D",
  title="AnnoTheia: A Semi-Automatic Annotation Toolkit for Audio-Visual Speech Technologies",
  booktitle="",
  volume="",
  number="",
  pages="",
  year="2024",
}
```

## <a name="license"></a> 📝 License
This work is protected by []()
