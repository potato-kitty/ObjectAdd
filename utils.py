import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm
import math
from torchviz import make_dot
import utils
from sklearn.cluster import KMeans
import os
from skimage import morphology
import copy
import torch.nn.functional as F
from random import choice

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

color_list = [
    [255, 0, 0],    
    [0, 255, 0],    
    [0, 0, 255],    
    [255, 255, 0],  
    [0, 255, 255],  
    [255, 0, 255],  
    [0, 0, 0],      
    [255, 255, 255],
    [255, 165, 0],  
    [128, 128, 128] 
]

def next_step(
    model,
    model_output: torch.FloatTensor,
    timestep: int,
    x: torch.FloatTensor,
    eta=0.,
    verbose=False
):
    """
    Inverse sampling for DDIM Inversion
    """
    if verbose:
        print("timestep: ", timestep)
    nxt_step = timestep
    timestep = min(timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps, 999)
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep] if timestep >= 0 else model.scheduler.final_alpha_cumprod
    alpha_prod_t_next = model.scheduler.alphas_cumprod[nxt_step]
    beta_prod_t = 1 - alpha_prod_t
    pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
    pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
    x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
    return x_next, pred_x0

@torch.no_grad()
def image2latent(model, image):
    DEVICE = device if torch.cuda.is_available() else torch.device("cpu")
    #if type(image) is Image:

    image = np.array(image)
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(0, 3, 1, 2).to(device)

    # input image density range [-1, 1]
    latents = model.vae.encode(image)['latent_dist'].mean
    latents = latents * 0.18215
    return latents

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def getX(self):
        return self.x
    def getY(self):
        return self.y

def kmeans_clustering(res,att_mask_here,input,star_x,star_y,end_x,end_y,n_clusters = 5, rate = 0.1, size_reduce = 20):
    cluster = KMeans(n_clusters=n_clusters,random_state=0).fit(input)
    result = cluster.labels_

    new_result = np.zeros([res,res,3])
    reshape_result = result.reshape(res,res)

    for i in range(res):
        for j in range(res):
            new_result[i,j,:] = color_list[reshape_result[i,j]]
    
    visual_img = new_result.astype(np.uint8)
    visual_img_show = Image.fromarray(visual_img).resize((512, 512))
    a = ImageDraw.ImageDraw(visual_img_show)
    visual_img_show = np.array(visual_img_show)


    result_tmp = (result.reshape(res,res) + 1)*att_mask_here
    num_list = []
    for i in range(n_clusters):
        num_list.append(np.sum(result_tmp == i+1)/len(result_tmp == i+1))
    
    selected_id = []
    max_value = max(num_list)
    #print("max_value: ",max_value)
    #max_id = num_list.index(max_value)
    for idx, rate_pixel in enumerate(num_list):
        if rate_pixel == max_value or rate_pixel > rate:
            selected_id.append(idx)
    for the_idx in range(len(result)):
        if result[the_idx] in selected_id:
            result[the_idx] = 1
        else:
            result[the_idx] = 0
    
    result = result.reshape(res,res)

    for i in range(res):
        for j in range(res):
            if result[i,j] != 1:
                new_result[i,j,:] = 0

    visual_img = new_result.astype(np.uint8)
    visual_img_show = Image.fromarray(visual_img).resize((512, 512))
    visual_img_show = np.array(visual_img_show)


    result = torch.tensor(mask_reshape(result,64))

    image = 255 * result / result.max()
    image = image.unsqueeze(-1).expand(*image.shape, 3)
    image = image.detach().numpy().astype(np.uint8)
    image = Image.fromarray(image).resize((512, 512))
    image = np.array(image)


    bool_result = np.array(result,dtype=bool)
    bool_result = morphology.remove_small_objects(bool_result,min_size=size_reduce)
    result = np.array(bool_result,dtype=np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(result, kernel)
    result = torch.tensor(cv2.dilate(erosion, kernel))

    image = 255 * result / result.max()
    image = image.unsqueeze(-1).expand(*image.shape, 3)
    image = image.detach().numpy().astype(np.uint8)
    image = Image.fromarray(image).resize((512, 512))
    image = np.array(image)

    return result.to(device)

def mask_reshape(original_mask,res):
    new_mask = np.zeros([res,res])
    att_mask_here = original_mask
    if att_mask_here.shape[0] < new_mask.shape[0]:
        ratio = int(new_mask.shape[0]/att_mask_here.shape[0])
        for i in range(att_mask_here.shape[0]):
            for j in range(att_mask_here.shape[1]):
                new_i = i*ratio
                new_j = j*ratio
                new_mask[new_i:new_i+ratio,new_j:new_j+ratio] = att_mask_here[i,j]
        att_mask_here = new_mask
    elif att_mask_here.shape[0] > new_mask.shape[0]:
        ratio = int(att_mask_here.shape[0]/new_mask.shape[0])
        for i in range(new_mask.shape[0]):
            for j in range(new_mask.shape[1]):
                new_i = i*ratio
                new_j = j*ratio
                new_mask[i,j] = att_mask_here[new_i,new_j]
    
    return new_mask
    

def get_dist(im, seed_location1, seed_location2, avg_val = None, alpha = 0.3):
    l1 = im[seed_location1.x, seed_location1.y]
    l2 = im[seed_location2.x, seed_location2.y]
    if avg_val is None:
        count = np.sqrt(np.sum(np.square(l1-l2)))
    else:
        count = (1 - alpha)*np.sqrt(np.sum(np.square(l1-l2))) + alpha*np.sqrt(np.sum(np.square(l2-avg_val)))
    return count

def regional_growth(im, seed_x = None, seed_y = None, T = 0.25, class_k = 1, seed_list = [],img_mark = None, mode = 1):
    connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0),
            Point(1, 1), Point(0, 1), Point(-1, 1), Point(-1, 0)]

    # import Image
    
    im_shape = im.shape
    height = im_shape[0]
    width = im_shape[1]

    if img_mark is None:
        img_mark = np.zeros([height, width])

    img_re = img_mark.cpu()

    if len(seed_list) == 0 and seed_x is not None and seed_y is not None:
        seed_list.append(Point(seed_x, seed_y))

    if mode == 0:
        avg_val = np.zeros(im.shape[-1])
        for seed in seed_list:
            avg_val += im[seed.x,seed.y]
        
        avg_val = avg_val/len(seed_list)

    while (len(seed_list) > 0):
        seed_tmp = seed_list[0]

        seed_list.pop(0)

        img_mark[seed_tmp.x, seed_tmp.y] = class_k

        if mode == 1:
            avg_val = im[seed_tmp.x, seed_tmp.y]
            caler = 1
            for i in range(8):
                tmp_x = seed_tmp.x + connects[i].x
                tmp_y = seed_tmp.y + connects[i].y
                if (tmp_x < 0 or tmp_y < 0 or tmp_x >= height or tmp_y >= width):
                    continue
                avg_val += im[tmp_x,tmp_y]
                caler = caler + 1
            avg_val = avg_val/9



        for i in range(8):
            tmpX = seed_tmp.x + connects[i].x
            tmpY = seed_tmp.y + connects[i].y

            if (tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= width):
                continue
            dist = get_dist(im, seed_tmp, Point(tmpX, tmpY),avg_val)
            #print("dist: ",dist)
            if (dist < T and img_mark[tmpX, tmpY] == 0):
                img_re[tmpX, tmpY] = 1
                img_mark[tmpX, tmpY] = class_k
                seed_list.append(Point(tmpX, tmpY))

    if seed_x is not None and seed_y is not None:
        img_re[seed_x,seed_y] = 1
    #print("img_re now: ",img_re)
    return img_re

def aggregate_attention(attention_store, res: int, from_where: List[str], is_cross: bool, select: int, batch_size:int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(batch_size, -1, res*res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

def Pharse2idx(prompt, phrases):
    phrases = [x.strip() for x in phrases.split(';')]
    prompt_list = prompt.strip('.').split(' ')
    object_positions = []
    for obj in phrases:
        obj_position = []
        for word in obj.split(' '):
            obj_first_index = prompt_list.index(word) + 1
            obj_position.append(obj_first_index)
        object_positions.append(obj_position)

    return object_positions

def compute_ca_loss(attn_maps_mid, attn_maps_up, bboxes, object_positions):
    loss = 0
    object_number = len(bboxes)
    if object_number == 0:
        return torch.tensor(0).float().to(device)
    # import pdb; pdb.set_trace()
    for attn_map_integrated in attn_maps_mid:
        attn_map = attn_map_integrated.chunk(2)[1].requires_grad_(True)

        #
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        for obj_idx in range(object_number):
            obj_loss = 0
            mask = torch.zeros(size=(H, W)).to(device)
            for obj_box in bboxes[obj_idx]:

                x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                    int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                mask[y_min: y_max, x_min: x_max] = 1

            for obj_position in object_positions[obj_idx]:
                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W).requires_grad_(True)

                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)

                obj_loss += torch.mean((1 - activation_value.requires_grad_(True)) ** 2)
            loss += (obj_loss/len(object_positions[obj_idx]))

    for attn_map_integrated in attn_maps_up[0]:
        attn_map = attn_map_integrated.chunk(2)[1].requires_grad_(True)
        #
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))

        for obj_idx in range(object_number):
            obj_loss = 0
            mask = torch.zeros(size=(H, W)).to(device)
            for obj_box in bboxes[obj_idx]:
                x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                    int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                mask[y_min: y_max, x_min: x_max] = 1

            for obj_position in object_positions[obj_idx]:
                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W).requires_grad_(True)

                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1) / ca_map_obj.reshape(b, -1).sum(dim=-1)

                obj_loss += torch.mean((1 - activation_value.requires_grad_(True)) ** 2)
            loss += (obj_loss / len(object_positions[obj_idx]))
    loss = loss / (object_number * (len(attn_maps_up[0]) + len(attn_maps_mid)))
    return loss

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, if_save = False, num_rows=1, offset_ratio=0.02,output_folder = None, img_name = None, box = None):
    if img_name is not None:
        name_list = ["/" + img_name + "_original_img","/" + img_name + "_edit_img","/" + img_name + "_edit_img_no_Kmeans"]
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]
    
    
    pil_img = Image.fromarray(image_)
    if if_save and output_folder is not None:
        for i in range(num_items):
            out_img = images[i]
            im = Image.fromarray(out_img)
            im.save(output_folder + name_list[i] + ".jpeg")

    display(pil_img)


def diffusion_step(DDP_model, controller, latents, context, t, guidance_scale, low_resource=False, if_tst_L = False):
    model = DDP_model
    if low_resource:
        noise_pred_uncond,_,_,_ = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text,_,_,_ = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred,_,_,_ = model.unet(latents_input, t, encoder_hidden_states=context)
        noise_pred = noise_pred["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    if if_tst_L and noise_pred.shape[0] > 2:
        noise_pred[1] = noise_pred[0]
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents 


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image

def latent2image_tensor(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image * 255)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(device)
    return latent, latents

def step_forward(latents,model,context,guidance_scale,t):
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
    noise_pred,_,_,_ = model.unet(latent_model_input, t, encoder_hidden_states=context)
    noise_pred = noise_pred.sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    latents = model.scheduler.step(noise_pred, t, latents).prev_sample

    return latents, noise_pred

def step_forward_noise(latents,model,context,guidance_scale,t):
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
    noise_pred,_,_,_ = model.unet(latent_model_input, t, encoder_hidden_states=context)
    noise_pred = noise_pred.sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    return noise_pred

def step_backward(model,latents,context,guidance_scale,t):

    model_inputs = torch.cat([latents] * 2)

    noise_pred,_,_,_ = model.unet(model_inputs, t, encoder_hidden_states=context)
    noise_pred = noise_pred.sample

    noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
    noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)

    return noise_pred

@torch.no_grad()
def invert(
    model,
    image,
    prompt,
    batch_size,
    num_inference_steps=50,
    guidance_scale=7.5,
    opt_invert=False,
    **kwds):
    """
    invert a real image into noise map with determinisc DDIM inversion
    """

    prompt = prompt[-1]
    batch_size = 1
    # text embeddings
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    # define initial latents
    latents = image2latent(model,image)
    start_latents = latents


    # unconditional embedding for classifier free guidance
    if guidance_scale > 1.:

        max_length = text_input.input_ids.shape[-1]
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

        context = [uncond_embeddings, text_embeddings]
        context = torch.cat(context)

    # interative sampling
    model.scheduler.set_timesteps(num_inference_steps)

    latents_list = [latents]
    or_latent_idx = 0.5
    inject_steps = 0.05 #0.05
    inject_len = 0.2 #0.2
    no_inject = 0
    inject_times = 0

    for i, t in enumerate(tqdm(reversed(model.scheduler.timesteps), desc="DDIM Inversion")):

        noise_pred = step_backward(model,latents,context,guidance_scale,t)
        noise_pred = noise_pred.requires_grad_(True)
        if not opt_invert:
            latents,_ = next_step(model, noise_pred, t, latents)
            if i > inject_steps*num_inference_steps and no_inject < inject_times:
                no_inject += 1
                latents = or_latent_idx*latents + (1 - or_latent_idx)*start_latents
            latents_list.append(latents)
        else:
            last_noise = noise_pred
            if (inject_steps + inject_len)*num_inference_steps > i > inject_steps*num_inference_steps:
                #and no_inject < inject_times:
                if i > 0:
                    latents = or_latent_idx*latents + (1 - or_latent_idx)*last_latent

            last_latent = latents
            latents,_ = next_step(model, noise_pred, t, latents)
            latents_list.append(latents)


    model.vae.to(device)
    latents_list.reverse()
    return latents, latents_list, start_latents

@torch.no_grad()
def text2image_ldm_stable(
    DDP_model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    real_img = None,
    full_img = None,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
    injection  = 0.1,
    is_switch = False,
    bboxes = None
):
    loss_scale = 30
    loss_threshold = 0.2
    max_iter = 10
    max_index_step = 10
    classifier_free_guidance = 7.5
    model = DDP_model
    batch_size = len(prompt)
    global device
    device = model.device

    register_attention_control(model, controller,one_input=True)

    if real_img is not None:
        inverse_img,resize_img_latents_list,resize_start_latents = invert(model,real_img,prompt,batch_size,opt_invert=True)
    if full_img is not None:
        inverse_img_full,_,_ = invert(model,full_img,prompt,batch_size,opt_invert=True)

    register_attention_control(model, controller,one_input=False)

    height = width = 512

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    if batch_size > 2:
        if real_img is not None:
            len_1 = len(prompt[0].split(' '))
            len_2 = len(prompt[1].split(' '))
        else:
            len_1 = len(prompt[0].split(' '))
            len_2 = len(prompt[-1].split(' '))
        print("len_1: ",len_1)
        print("len_2: ",len_2)
        controller.object_location = len_1 + len_2

    if batch_size > 2:
        if real_img is not None:
            tmp_emb = torch.clone(text_embeddings[1])
            text_embeddings[1][0] = text_embeddings[-1][0]
            text_embeddings[1][1:len_1 + 1] = text_embeddings[0][1:len_1 + 1]
            text_embeddings[1][len_1+1:len_1 + len_2 + 1] = tmp_emb[1:1 + len_2]
            text_embeddings[1][len_1 + len_2 + 1:] = text_embeddings[-1][len_1 + len_2 + 1:]
        else:
            text_embeddings[1][0] = text_embeddings[-1][0]
            text_embeddings[1][1:len_1 + 1] = text_embeddings[0][1:len_1 + 1]
            text_embeddings[1][len_1+1:len_1 + len_2 + 1] = text_embeddings[-1][1:1 + len_2]
            text_embeddings[1][len_1 + len_2 + 1:] = text_embeddings[-1][len_1 + len_2 + 1:]

    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)

    if batch_size > 2 and controller.mask is not None:
        new_mask = np.zeros([64,64])
        att_mask = controller.mask
        if att_mask.shape[0] < new_mask.shape[0]:
            ratio = int(new_mask.shape[0]/att_mask.shape[0])
            for i in range(att_mask.shape[0]):
                for j in range(att_mask.shape[1]):
                    new_i = i*ratio
                    new_j = j*ratio
                    new_mask[new_i:new_i+ratio,new_j:new_j+ratio] = att_mask[i,j]
            att_mask = new_mask
        elif att_mask.shape[0] > new_mask.shape[0]:
            ratio = int(att_mask.shape[0]/new_mask.shape[0])
            for i in range(new_mask.shape[0]):
                for j in range(new_mask.shape[1]):
                    new_i = i*ratio
                    new_j = j*ratio
                    new_mask[i,j] = att_mask[new_i,new_j]
            att_mask = new_mask
        
        att_mask = torch.tensor(att_mask).to(device)

    if real_img is not None:
        if full_img is None:
            latents[-1] = inverse_img[-1]
        else:
            latents[-1] = inverse_img_full[-1]
    
    # set timesteps
    model.scheduler.set_timesteps(num_inference_steps)
    max_time = model.scheduler.timesteps[0].numpy()


    if batch_size > 2:
        latents = latents * model.scheduler.init_noise_sigma

        loss = torch.tensor(10000)
        if real_img is not None:
            last_list = prompt[1].split(' ')
            phrases = last_list[-1]
            object_positions = Pharse2idx(prompt[1]+'.', phrases)
        else:
            last_list = prompt[-1].split(' ')
            phrases = last_list[-1]
            object_positions = Pharse2idx(prompt[-1]+'.', phrases)

    if real_img is None and full_img is None:
        ly_cntrl = True
    else:
        ly_cntrl = False

    lantent_inject_t = 0.8
    lantent_inject_t_min = 0.7


    no_inject = True
    if batch_size > 2 and bboxes is not None and lantent_inject_t < 1 and max_iter > 0 and ly_cntrl:
        print("with layout control!")
    else:
        print("without layout control!")
    
    for i, t in enumerate(tqdm(model.scheduler.timesteps)):
        ctrled = False
        if batch_size > 2 and (t > lantent_inject_t*max_time) and ly_cntrl and bboxes is not None:
            iteration = 0
            #prev_counter = model.scheduler.counter
            while loss.item() / loss_scale > loss_threshold and iteration < max_iter and i < max_index_step:
                with torch.enable_grad():
                    
                    latent_model_input = latents[-1].requires_grad_(True) 
                    latent_model_input = latent_model_input.unsqueeze(0)
                    latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
                    
                    _, attn_map_integrated_up, attn_map_integrated_mid, _ = model.unet(latent_model_input, t, encoder_hidden_states=text_embeddings[-1])

                    # update latents with guidance
                    
                    loss = compute_ca_loss(attn_map_integrated_mid, attn_map_integrated_up, bboxes=bboxes,
                                        object_positions=object_positions) * loss_scale

                    loss = loss.requires_grad_(True)

                    grad_cond = torch.autograd.grad(loss,latent_model_input)[0]

                    latents[-1] = latents[-1] - grad_cond*1.5

                    iteration += 1
                    torch.cuda.empty_cache()

            with torch.no_grad():
                latent_model_input = torch.cat([latents] * 2)

                latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
                noise_pred,_,_,_ = model.unet(latent_model_input, t, encoder_hidden_states=context)
                noise_pred = noise_pred.sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + classifier_free_guidance * (noise_pred_text - noise_pred_uncond)

                latents = model.scheduler.step(noise_pred, t, latents).prev_sample
                latents[1] = latents[-1]*att_mask + latents[0]*(1-att_mask)
                torch.cuda.empty_cache()
                ctrled = True

        elif batch_size > 2 and (lantent_inject_t*max_time > t > lantent_inject_t_min*max_time) and real_img is not None:
            new_latents = torch.clone(latents)
            if no_inject:
                new_latents[1] = new_latents[1]*(1 - att_mask) + resize_img_latents_list[i]*att_mask
                no_inject = False
            latents[1] = new_latents[1]*att_mask + latents[0]*(1-att_mask)

        if not ctrled:
            latents = diffusion_step(DDP_model, controller, latents, context, t, guidance_scale, low_resource=low_resource)
        
        res = 16
        if_cross = True
        concated = False
        
        if batch_size > 2 and t < 0.2*max_time and not concated and controller.mask is not None:
            att_mask_here = mask_reshape(controller.mask,res)
            select = 1
            attention_maps = aggregate_attention(controller, res, ["up"], if_cross, select, batch_size)

            cross_attn_obj = attention_maps
            
            concated = True

            att_mask_512 = mask_reshape(att_mask_here,512)
            star_x = 0
            star_y = 0
            end_x = 0
            end_y = 0
            for i in range(att_mask_512.shape[0]):
                for j in range(att_mask_512.shape[1]):
                    if att_mask_512[i,j] == 1:
                        if star_x == 0 and star_y == 0:
                            star_x = i
                            star_y = j
                        else:
                            end_x = i
                            end_y = j

            result = kmeans_clustering(res,att_mask_here,cross_attn_obj,star_x,star_y,end_x,end_y,n_clusters = 6, rate = 0.3)           

            out_img = result
            next_list = []
            for i in range(out_img.shape[0]):
                for j in range(out_img.shape[1]):
                    if out_img[i,j] == 1:
                        next_list.append(Point(i, j))

            latent_img = latents[1].permute(1,2,0)
            latent_img = latent_img.cpu().detach().numpy().astype(np.float64)
            mask_latent = regional_growth(latent_img,seed_list = next_list,img_mark = out_img, T = 5,mode=1)
            
            bool_result = np.array(mask_latent,dtype=bool)
            bool_result = morphology.remove_small_objects(bool_result,min_size=20,connectivity=2)
            mask_latent = np.array(bool_result,dtype=np.uint8)
            
            out_mask = torch.tensor(mask_latent)

            image = 255 * out_mask / out_mask.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.cpu().detach().numpy().astype(np.uint8)
            image = Image.fromarray(image).resize((512, 512))
            image = np.array(image)

            out_mask = out_mask.cpu().detach().numpy().astype(np.uint8)
            out_mask = torch.tensor(np.array(Image.fromarray(out_mask).resize((64, 64)))).to(device)
            latents[1] = latents[0]*(1 - out_mask) + latents[1]*out_mask
 
    image = latent2image(model.vae, latents)

    return image, latent


def register_attention_control(model, controller, no_switch = False, one_input = False):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None, one = one_input):
            batch_size, sequence_length, dim = x.shape
            if context is not None and one:
                context = context[-2:]
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim

            #
            if len(context.shape) < 3:
                context = context.unsqueeze(0)
            if context.shape[0] > 2:
                    attn = controller(attn, is_cross, place_in_unet)

            attn = attn.softmax(dim=-1)

            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)
            out = to_out(out)
            torch.cuda.empty_cache()
            return out, attn

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count
