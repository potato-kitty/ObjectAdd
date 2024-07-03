from typing import Union, Tuple, List, Dict
import torch
import torch.nn.functional as nnf
import numpy as np
import abc
import utils

LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77


class LocalBlend:

    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        self.alpha_layers = self.alpha_layers.to(maps.device)
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words, threshold=.3, tokenizer = None):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.threshold = threshold
        self.alpha_layers = alpha_layers


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str, no_switch = False):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if self.no_switch:
            return attn
        
        h = attn.shape[0] // (self.batch_size)
        attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
        attn_base, attn_mid, attn_repalce = attn[0].clone(), attn[1].clone(), attn[1:].clone()

        att_mask = self.mask
        size_msk = int(attn_mid.shape[-2]**0.5)
        if att_mask is not None:
            ratio_img = int(att_mask.shape[0]/size_msk)

        x = 15 
        y = 164 
        x_w = 100
        y_h = 191

        if att_mask is not None:
            new_mask = np.zeros([size_msk,size_msk])
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
            
            att_mask = torch.tensor(att_mask).to(attn.device)

            x = round(x/ratio_img)
            y = round(y/ratio_img) 
            x_w = round(x_w/ratio_img)
            y_h = round(y_h/ratio_img)

        scaler = 0.75
        min_scaler = 0.65
        
        if is_cross and (int(scaler * self.num_steps) > self.cur_step > int(min_scaler * self.num_steps)) and att_mask is not None:

            att_mask = torch.unsqueeze(att_mask,0)

            idx = self.object_location

            att_mask = att_mask.repeat(8,1,1)

            attn[1][:,:,idx] = (att_mask).reshape(attn[1][:,:,idx].shape)

        elif (not is_cross) and att_mask is not None:
            apply_process = False
            if (int(scaler * self.num_steps) > self.cur_step > int(min_scaler * self.num_steps)):
                apply_process = True
            apply_process = False
            
            if apply_process:

                selected_idx = []
                # here we assume that the gennerated image MUST be squar
                
                new_mask = np.zeros([attn_mid.shape[-2],attn_mid.shape[-1]])
                new_mask = torch.tensor(new_mask).to(attn.device)
                
                for i in range(att_mask.shape[0]):
                    for j in range(att_mask.shape[1]):
                        if att_mask[i,j] > 0:
                            selected_idx.append(i*att_mask.shape[0] + j)
                            new_mask[i*att_mask.shape[0] + j,:] = att_mask[i,j]
                            new_mask[:,i*att_mask.shape[1] + j] = att_mask[i,j]
                att_mask = new_mask

                for i in range(attn[1].shape[0]):
                    value = max(torch.mean(attn_base[i,:,:]),1)**2
                    attn[1][i,:,:] = (att_mask*value + attn_base[i,:,:])

        
        attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend=None,where_change = None, mask = None, full_mask = None, no_switch = False):
        super(AttentionControlEdit, self).__init__()
        self.copy_map = False
        self.if_sqr_replace = False
        self.where_change = where_change
        self.mask = mask
        self.full_mask = full_mask
        self.show = True
        self.object_location = 0
        self.num_steps = num_steps
        self.no_switch = no_switch
        self.batch_size = len(prompts)
        if type(self_replace_steps) is float:
            self_replace_steps = 0 , self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def change_sqr_replace(self,to_what):
        self.if_sqr_replace = to_what

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, where_change: bool = None, mask : torch.tensor =None,
                 full_mask : torch.tensor =None,local_blend = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, where_change, mask, full_mask)
