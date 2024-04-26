# ObjectAdd
The codes of our paper ObjectAdd: Adding Objects into Image via a Training-Free Diffusion Modification Fashion

We are still preparing our code, once it is done, we will upload it here.

0. To try our example, just click Run All in "main_control.ipynb". Or, to apply our approach in your on image and mask, tthe whole procedure is as following: Firstly run "main_control.ipynb" untill the mark down bolck with hint "Run untill here to genenrate the original image", set if_save parameter to True for saving the generated image and change the prompt for generating different images. Then use "choose_object.py" to load generated image and draw mask on it. Finally, run the rest part of "main_control.ipynb" to load the mask and obtain the final results.  

1. If you got problem for download the SD pre-train model by "StableDiffusionPipeline.from_pretrained", download it by yourself and place it under "CompVis" folder.

2. Before running the codes, copy three files in folder "replace_diffusers" to ".../site-packages/diffusers/models/" and replace the original files. Rember to make a backup of these replaced files incase you want to run other project in this enviroment.
