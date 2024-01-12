<h1 align="left"> 📜 How to incorporate a new model into Annotheia</h1>

In this tutorial, we are going to learn how to incorporate a new model into the toolkit to replace the model used in one of its modules with another. There may be different reasons for this, from the need for a specific ASR model for our language to the fact that said model is more precise or efficient. **In addition,** you will be able to infer how to include a new step into our pipeline, such as a new body landmarker module!

## 🏗️ How is AnnoTheia structured?

<div align="center"> <img src="../doc/image/annotheia_architecture.png" width=712> </div>

First of all, it is advisable to know a little more about the toolkit and how it is organized:

- The first thing we need is a [configuration file](../configs/annotheia_pipeline_spanish.yaml) that includes the settings to build each module composing the pipeline, as well as different pipeline hyper-parameters.
- When running the [main_scenes.py](../main_scenes.py#L36) script, we create an instance of the `DetectCandidateScenesTask` class.
- This `DetectCandidateScenesTask` object, defined in [detect_candidate_scenes_task.py](../tasks/detect_candidate_scenes_task.py#L77C7-L77C32), is in charge of building and setting up all the modules specified in the configuration file, such as the scene detector, face detector, and so on.
- Once all the modules have been independently built, we can conform our pipeline by instantiating the `Pipeline` class, whose definition you can find in [annotheia_pipeline.py](https://github.com/joactr/AnnoTheia/blob/david-branch/pipelines/annotheia_pipeline.py#L24).
- Finally, the `Pipeline` processes the video to detect the candidate scenes for the future database.

A question arises: **how is the flexibility that the toolkit offers in terms of changing or including a new model or module possible?** If you inspect this directory, you will see that for each module and for each model that can represent it, there is a folder. If you look again, you will see that in each module directory, there exists a script like `abs_{name_of_the_module}.py`. These scripts define the different **abstracts classes** we need to force your models to receive and output the data as the pipeline expects. By following these guidelines, regardless of which model you include, the pipeline will continue to work 🪄!

## 💪 Hands on!

Imagine we want to incorporate the **face aligner** of [Google Mediapipe](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) because we think it is a better option for our project, for example. Let's go step by step:

#### 1. Placing our New Model

In order to maintain the toolkit's structure, we will create the following directory as if we were at the root directory of the repository:

```
cd ./modules/face_alignment/
mkdir ./mediapipe_face_aligner/
cd ./mediapipe_face_aligner/
```

#### 2. Installation & Model Download

According to the [MediaPipe's tutorial](https://github.com/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb):

- We will install MediaPipe:
```
pip install mediapipe
```
- We will download the off-the-shelf model bundle:

```
mkdir ./models/
wget -O ./models/face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

#### 3. Creating the Model Class

Creating an class for our model in the `mediapipe_face_aligner.py` script. Note that our face aligner will inherit the abstract class defined in [../abs_face_aligner.py](../abs_face_aligner.py):

```python

import mediapipe as mp
from termcolor import cprint
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from modules.face_alignment.abs_face_aligner import AbsFaceAligner

class MediaPipeFaceAligner(AbsFaceAligner):
    def __init__(self, model_asset_path="./models/face_landmarker_v2_with_blendshapes.task"):

        # -- creating model configuration
        base_options = python.BaseOptions(
            model_asset_path=model_asset_path,
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        # -- building face aligner
        self.face_aligner = vision.FaceLandmarker.create_from_options(options)

        # -- printing useful information for the user
        cprint(
            f"\t(Face Alignment) MediaPipe Face Aligner loaded from ({model_asset_path})",
            "cyan", attrs=["bold", "reverse"],
        )

    def detect_facial_landmarks(self, frame, face_bbs):
        """Detect the facial landmarks of each face appearing on the frame.
        Args:
            frame (np.ndarray): a frame read from the scene clip.
            face_bbs (np.ndarray): face bounding boxes provided by the face detector.
        Returns:
            np.ndarray: array containing the detected facial landmarks (N,L,2),
          where N refers to number of faces and L to the number of detected landmarks.
        """

        # -- in this case, we do not need the face bounding boxes
        landmarks = self.face_aligner.detect(frame).face_landmarks
        return landmarks
```

#### 4. Importing Issues

Just to make it easier to import this model in the future, we will include it in the [../\_\_init\_\_.py](../__init__.py):

```
from modules.face_alignment.mediapipe_face_aligner.mediapipe_face_aligner import MediaPipeFaceAligner
```

#### 5. Adding the New Model to the Task

Once our model has been created, we can come back to the root directory of the repository. Then, we will include our new model into the setting up of our [./tasks/detect_candidate_scenes_task.py](../tasks/detect_candidate_scenes_task.py) script:

- Include it to allow its definition by means of the configuration file:

```
face_alignment_choices = ClassChoices(
    name="face_alignment",
    classes=dict(
        fan=FANAligner,
        mediapipe=Media
    ),
    type_check=AbsFaceAligner,
    default=None,
    optional=True
)
```

