import os
import torch
import collections
import torch.nn as nn
from functools import partial
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, PNDMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from models.unet_2d_condition import UNet2DConditionModel
from utils.attention_utils import CrossAttentionLayers, SelfAttentionLayers

# suppress partial model loading warning
logging.set_verbosity_error()


class RegionDiffusion(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device
        self.num_train_timesteps = 1000
        self.clip_gradient = False

        print(f'[INFO] loading stable diffusion...')
        model_id = 'runwayml/stable-diffusion-v1-5'

        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder='text_encoder').to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet").to(self.device)

        self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                       num_train_timesteps=self.num_train_timesteps, skip_prk_steps=True, steps_offset=1)
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        self.masks = []
        self.attention_maps = None
        self.selfattn_maps = None
        self.crossattn_maps = None
        self.color_loss = torch.nn.functional.mse_loss
        self.forward_hooks = []
        self.forward_replacement_hooks = []

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length',
                                      max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def get_text_embeds_list(self, prompts):
        # prompts: [list]
        text_embeddings = []
        for prompt in prompts:
            # Tokenize text and get embeddings
            text_input = self.tokenizer(
                [prompt], padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

            with torch.no_grad():
                text_embeddings.append(self.text_encoder(
                    text_input.input_ids.to(self.device))[0])

        return text_embeddings

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5,
                        latents=None, use_guidance=False, text_format_dict={}, inject_selfattn=0, inject_background=0):

        if latents is None:
            latents = torch.randn(
                (1, self.unet.in_channels, height // 8, width // 8), device=self.device)

        if inject_selfattn > 0 or inject_background > 0:
            latents_reference = latents.clone().detach()
        self.scheduler.set_timesteps(num_inference_steps)
        n_styles = text_embeddings.shape[0]-1
        assert n_styles == len(self.masks)
        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):

                # predict the noise residual
                with torch.no_grad():
                    # tokens without any attributes
                    feat_inject_step = t > (1-inject_selfattn) * 1000
                    background_inject_step = i == int(inject_background * len(self.scheduler.timesteps)) and inject_background > 0
                    noise_pred_uncond_cur = self.unet(latents, t, encoder_hidden_states=text_embeddings[:1],
                                                     )['sample']
                    self.register_fontsize_hooks(text_format_dict)
                    noise_pred_text_cur = self.unet(latents, t, encoder_hidden_states=text_embeddings[-1:],
                                                    )['sample']
                    self.remove_fontsize_hooks()
                    if inject_selfattn > 0 or inject_background > 0:
                        noise_pred_uncond_refer = self.unet(latents_reference, t, encoder_hidden_states=text_embeddings[:1],
                                                            )['sample']
                        self.register_selfattn_hooks(feat_inject_step)
                        noise_pred_text_refer = self.unet(latents_reference, t, encoder_hidden_states=text_embeddings[-1:],
                                                          )['sample']
                        self.remove_selfattn_hooks()
                    noise_pred_uncond = noise_pred_uncond_cur * self.masks[-1]
                    noise_pred_text = noise_pred_text_cur * self.masks[-1]
                    # tokens with attributes
                    for style_i, mask in enumerate(self.masks[:-1]):
                        self.register_replacement_hooks(feat_inject_step)
                        noise_pred_text_cur = self.unet(latents, t, encoder_hidden_states=text_embeddings[style_i+1:style_i+2],
                                                        )['sample']
                        self.remove_replacement_hooks()
                        noise_pred_uncond = noise_pred_uncond + noise_pred_uncond_cur*mask
                        noise_pred_text = noise_pred_text + noise_pred_text_cur*mask
                
                # perform classifier-free guidance
                noise_pred = noise_pred_uncond + guidance_scale * \
                    (noise_pred_text - noise_pred_uncond)

                if inject_selfattn > 0 or inject_background > 0:
                    noise_pred_refer = noise_pred_uncond_refer + guidance_scale * \
                        (noise_pred_text_refer - noise_pred_uncond_refer)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_reference = self.scheduler.step(torch.cat([noise_pred, noise_pred_refer]), t,
                                                            torch.cat([latents, latents_reference]))[
                        'prev_sample']
                    latents, latents_reference = torch.chunk(
                        latents_reference, 2, dim=0)

                else:
                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents)[
                        'prev_sample']

                # apply guidance
                if use_guidance and t < text_format_dict['guidance_start_step']:
                    with torch.enable_grad():
                        if not latents.requires_grad:
                            latents.requires_grad = True
                        latents_0 = self.predict_x0(latents, noise_pred, t)
                        latents_inp = 1 / 0.18215 * latents_0
                        imgs = self.vae.decode(latents_inp).sample
                        imgs = (imgs / 2 + 0.5).clamp(0, 1)
                        loss_total = 0.
                        for attn_map, rgb_val in zip(text_format_dict['color_obj_atten'], text_format_dict['target_RGB']):
                            avg_rgb = (
                                imgs*attn_map[:, 0]).sum(2).sum(2)/attn_map[:, 0].sum()
                            loss = self.color_loss(
                                avg_rgb, rgb_val[:, :, 0, 0])*100
                            loss_total += loss
                        loss_total.backward()
                    latents = (
                        latents - latents.grad * text_format_dict['color_guidance_weight'] * text_format_dict['color_obj_atten_all']).detach().clone()

                # apply background injection
                if background_inject_step:
                    latents = latents_reference * self.masks[-1] + latents * \
                        (1-self.masks[-1])
        return latents

    def predict_x0(self, x_t, eps_t, t):
        alpha_t = self.scheduler.alphas_cumprod[t]
        return (x_t - eps_t * torch.sqrt(1-alpha_t)) / torch.sqrt(alpha_t)

    def produce_attn_maps(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50,
                          guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeddings = self.get_text_embeds(
            prompts, negative_prompts)  # [2, 77, 768]
        if latents is None:
            latents = torch.randn(
                (text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)
        self.remove_replacement_hooks()

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * \
                    (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)[
                    'prev_sample']

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50,
                      guidance_scale=7.5, latents=None, text_format_dict={}, use_guidance=False, inject_selfattn=0, inject_background=0):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(
            prompts, negative_prompts)  # [2, 77, 768]

        # else:
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents,
                                       num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                                       use_guidance=use_guidance, text_format_dict=text_format_dict,
                                       inject_selfattn=inject_selfattn, inject_background=inject_background)  # [1, 4, 64, 64]
        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs

    def reset_attention_maps(self):
        r"""Function to reset attention maps.
        We reset attention maps because we append them while getting hooks
        to visualize attention maps for every step.
        """
        for key in self.selfattn_maps:
            self.selfattn_maps[key] = []
        for key in self.crossattn_maps:
            self.crossattn_maps[key] = []

    def register_evaluation_hooks(self):
        r"""Function for registering hooks during evaluation.
        We mainly store activation maps averaged over queries.
        """
        self.forward_hooks = []

        def save_activations(activations, name, module, inp, out):
            r"""
            PyTorch Forward hook to save outputs at each forward pass.
            """
            # out[0] - final output of attention layer
            # out[1] - attention probability matrix
            if 'attn2' in name:
                assert out[1].shape[-1] == 77
                activations[name].append(out[1].detach().cpu())
            else:
                assert out[1].shape[-1] != 77
        attention_dict = collections.defaultdict(list)
        for name, module in self.unet.named_modules():
            leaf_name = name.split('.')[-1]
            if 'attn' in leaf_name:
                # Register hook to obtain outputs at every attention layer.
                self.forward_hooks.append(module.register_forward_hook(
                    partial(save_activations, attention_dict, name)
                ))
        # attention_dict is a dictionary containing attention maps for every attention layer
        self.attention_maps = attention_dict

    def register_selfattn_hooks(self, feat_inject_step=False):
        r"""Function for registering hooks during evaluation.
        We mainly store activation maps averaged over queries.
        """
        self.selfattn_forward_hooks = []

        def save_activations(activations, name, module, inp, out):
            r"""
            PyTorch Forward hook to save outputs at each forward pass.
            """
            # out[0] - final output of attention layer
            # out[1] - attention probability matrix
            if 'attn2' in name:
                assert out[1][1].shape[-1] == 77
                # cross attention injection
                # activations[name] = out[1][1].detach()
            else:
                assert out[1][1].shape[-1] != 77
                activations[name] = out[1][1].detach()

        def save_resnet_activations(activations, name, module, inp, out):
            r"""
            PyTorch Forward hook to save outputs at each forward pass.
            """
            # out[0] - final output of residual layer
            # out[1] - residual hidden feature
            assert out[1].shape[-1] == 16
            activations[name] = out[1].detach()
        attention_dict = collections.defaultdict(list)
        for name, module in self.unet.named_modules():
            leaf_name = name.split('.')[-1]
            if 'attn' in leaf_name and feat_inject_step:
                # Register hook to obtain outputs at every attention layer.
                self.selfattn_forward_hooks.append(module.register_forward_hook(
                    partial(save_activations, attention_dict, name)
                ))
            if name == 'up_blocks.1.resnets.1' and feat_inject_step:
                self.selfattn_forward_hooks.append(module.register_forward_hook(
                    partial(save_resnet_activations, attention_dict, name)
                ))
        # attention_dict is a dictionary containing attention maps for every attention layer
        self.self_attention_maps_cur = attention_dict

    def register_replacement_hooks(self, feat_inject_step=False):
        r"""Function for registering hooks to replace self attention.
        """
        self.forward_replacement_hooks = []

        def replace_activations(name, module, args):
            r"""
            PyTorch Forward hook to save outputs at each forward pass.
            """
            if 'attn1' in name:
                modified_args = (args[0], self.self_attention_maps_cur[name])
                return modified_args
                # cross attention injection
            # elif 'attn2' in name:
            #     modified_map = {
            #         'reference': self.self_attention_maps_cur[name],
            #         'inject_pos': self.inject_pos,
            #     }
            #     modified_args = (args[0], modified_map)
            #     return modified_args

        def replace_resnet_activations(name, module, args):
            r"""
            PyTorch Forward hook to save outputs at each forward pass.
            """
            modified_args = (args[0], args[1],
                             self.self_attention_maps_cur[name])
            return modified_args
        for name, module in self.unet.named_modules():
            leaf_name = name.split('.')[-1]
            if 'attn' in leaf_name and feat_inject_step:
                # Register hook to obtain outputs at every attention layer.
                self.forward_replacement_hooks.append(module.register_forward_pre_hook(
                    partial(replace_activations, name)
                ))
            if name == 'up_blocks.1.resnets.1' and feat_inject_step:
                # Register hook to obtain outputs at every attention layer.
                self.forward_replacement_hooks.append(module.register_forward_pre_hook(
                    partial(replace_resnet_activations, name)
                ))

    def register_tokenmap_hooks(self):
        r"""Function for registering hooks during evaluation.
        We mainly store activation maps averaged over queries.
        """
        self.forward_hooks = []

        def save_activations(selfattn_maps, crossattn_maps, n_maps, name, module, inp, out):
            r"""
            PyTorch Forward hook to save outputs at each forward pass.
            """
            # out[0] - final output of attention layer
            # out[1] - attention probability matrices
            if name in n_maps:
                n_maps[name] += 1
            else:
                n_maps[name] = 1
            if 'attn2' in name:
                assert out[1][0].shape[-1] == 77
                if name in CrossAttentionLayers and n_maps[name] > 10:
                    if name in crossattn_maps:
                        crossattn_maps[name] += out[1][0].detach().cpu()[1:2]
                    else:
                        crossattn_maps[name] = out[1][0].detach().cpu()[1:2]
            else:
                assert out[1][0].shape[-1] != 77
                if name in SelfAttentionLayers and n_maps[name] > 10:
                    if name in crossattn_maps:
                        selfattn_maps[name] += out[1][0].detach().cpu()[1:2]
                    else:
                        selfattn_maps[name] = out[1][0].detach().cpu()[1:2]

        selfattn_maps = collections.defaultdict(list)
        crossattn_maps = collections.defaultdict(list)
        n_maps = collections.defaultdict(list)

        for name, module in self.unet.named_modules():
            leaf_name = name.split('.')[-1]
            if 'attn' in leaf_name:
                # Register hook to obtain outputs at every attention layer.
                self.forward_hooks.append(module.register_forward_hook(
                    partial(save_activations, selfattn_maps,
                            crossattn_maps, n_maps, name)
                ))
        # attention_dict is a dictionary containing attention maps for every attention layer
        self.selfattn_maps = selfattn_maps
        self.crossattn_maps = crossattn_maps
        self.n_maps = n_maps

    def remove_tokenmap_hooks(self):
        for hook in self.forward_hooks:
            hook.remove()
        self.selfattn_maps = None
        self.crossattn_maps = None
        self.n_maps = None

    def remove_evaluation_hooks(self):
        for hook in self.forward_hooks:
            hook.remove()
        self.attention_maps = None

    def remove_replacement_hooks(self):
        for hook in self.forward_replacement_hooks:
            hook.remove()

    def remove_selfattn_hooks(self):
        for hook in self.selfattn_forward_hooks:
            hook.remove()

    def register_fontsize_hooks(self, text_format_dict={}):
        r"""Function for registering hooks to replace self attention.
        """
        self.forward_fontsize_hooks = []

        def adjust_attn_weights(name, module, args):
            r"""
            PyTorch Forward hook to save outputs at each forward pass.
            """
            if 'attn2' in name:
                modified_args = (args[0], None, attn_weights)
                return modified_args

        if 'word_pos' in text_format_dict and text_format_dict['word_pos'] is not None \
            and 'font_size' in text_format_dict and text_format_dict['font_size'] is not None:
            attn_weights = {'word_pos': text_format_dict['word_pos'], 'font_size': text_format_dict['font_size']}
        else:
            attn_weights = None

        for name, module in self.unet.named_modules():
            leaf_name = name.split('.')[-1]
            if 'attn' in leaf_name and attn_weights is not None:
                # Register hook to obtain outputs at every attention layer.
                self.forward_fontsize_hooks.append(module.register_forward_pre_hook(
                    partial(adjust_attn_weights, name)
                ))

    def remove_fontsize_hooks(self):
        for hook in self.forward_fontsize_hooks:
            hook.remove()