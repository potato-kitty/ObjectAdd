# ObjectAdd
This is the official implementation of our paper ObjectAdd: Adding Objects into Image via a Training-Free Diffusion Modification Fashion

0. To try our example, run by_reference.ipynb

1. If you got problem for download the SD pre-train model by "StableDiffusionPipeline.from_pretrained", download it by yourself and place it under "CompVis" folder.

2. Before running the codes, copy three files in folder "replace_diffusers" to ".../site-packages/diffusers/models/" and replace the original files. Rember to make a backup of these replaced files incase you want to run other project in this enviroment.

3. Your can prepare your own data following the form of txt files in "mask_info" folder, the first line of these folder is the x-corordinate of left-top point of the drawn box, the secound one is its y-coordinate, and following with the width and height, the last line is the prompt of object you want to add. The defult setting of our codes requires the object word should be in the last position, for example, to add a running cat, you should type the prompt as 'A runing cat' instead of 'A cat runing'.
