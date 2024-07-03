# ObjectAdd
This is the official implementation of our paper ObjectAdd: Adding Objects into Image via a Training-Free Diffusion Modification Fashion

0. To try our example, run by_reference.ipynb

1. If you got problem for download the SD pre-train model by "StableDiffusionPipeline.from_pretrained", download it by yourself and place it under "CompVis" folder.

2. Before running the codes, copy three files in folder "replace_diffusers" to ".../site-packages/diffusers/models/" and replace the original files. Rember to make a backup of these replaced files incase you want to run other project in this enviroment.

3. Your can prepare your own data following the form of txt files in "mask_info" folder, the first line of these folder is the x-corordinate of left-top point of the drawn box, the secound one is its y-coordinate, and following with the width and height, the last line is the prompt of object you want to add. The defult setting of our codes requires the object word should be in the last position, for example, to add a running cat, you should type the prompt as 'A runing cat' instead of 'A cat runing'.

#Reference
Part of our codes are based on following two projects:
1. [prompt-to-prompt](https://github.com/google/prompt-to-prompt)
@article{hertz2022prompt,
  title = {Prompt-to-Prompt Image Editing with Cross Attention Control},
  author = {Hertz, Amir and Mokady, Ron and Tenenbaum, Jay and Aberman, Kfir and Pritch, Yael and Cohen-Or, Daniel},
  journal = {arXiv preprint arXiv:2208.01626},
  year = {2022},
}
2. [layout-guidance](https://github.com/silent-chen/layout-guidance)
@article{chen2023trainingfree,
  title={Training-Free Layout Control with Cross-Attention Guidance}, 
  author={Minghao Chen and Iro Laina and Andrea Vedaldi},
  journal={arXiv preprint arXiv:2304.03373},
  year={2023}
}
