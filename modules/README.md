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
