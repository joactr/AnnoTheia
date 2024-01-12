<h1 align="left"> 📜 How to incorporate a new model into Annotheia</h1>

In this tutorial, we are going to learn how to incorporate a new model into the toolkit to replace the model used in one of its modules with another. There may be different reasons for this, from the need for a specific ASR model for our language to the fact that said model is more precise or efficient. **In addition,** you will be able to infer how to include a new step into our pipeline, such as a new body landmarker module!

## 🏗️ How is AnnoTheia structured?

<div align="center"> <img src="../doc/image/annotheia_architecture.png" width=712> </div>

First of all, it is advisable to know a little more about the tool and how it is organized:

- The first thing we need is a [configuration file](../configs/annotheia_pipeline_spanish.yaml) defining the different settings to build each module composing the pipeline.
- When running the [main_scenes.py](../main_scenes.py#L36) script, we create an instance of the DetectCandidateScenesTask object, defined in [detect_candidate_scenes_task.py](../tasks/detect_candidate_scenes_task.py#L77C7-L77C32). Concretely, this object will build and set up all the modules specified in the configuration file.
- 
