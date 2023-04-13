import os
import torch
import collections
import torch.nn as nn
from functools import partial
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, PNDMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from models.unet_2d_condition import UNet2DConditionModel

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
        self.color_loss = torch.nn.functional.mse_loss

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
                        latents=None, use_grad_guidance=False, text_format_dict={}):

        if latents is None:
            latents = torch.randn(
                (1, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)
        n_styles = text_embeddings.shape[0]-1
        assert n_styles == len(self.masks)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):

                # predict the noise residual
                with torch.no_grad():
                    noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=text_embeddings[:1],
                                                  text_format_dict={})['sample']
                    noise_pred_text = None
                    for style_i, mask in enumerate(self.masks):
                        if style_i < len(self.masks) - 1:
                            masked_latent = latents
                            noise_pred_text_cur = self.unet(masked_latent, t, encoder_hidden_states=text_embeddings[style_i+1:style_i+2],
                                                            text_format_dict={})['sample']
                        else:
                            noise_pred_text_cur = self.unet(latents, t, encoder_hidden_states=text_embeddings[style_i+1:style_i+2],
                                                            text_format_dict=text_format_dict)['sample']
                        if noise_pred_text is None:
                            noise_pred_text = noise_pred_text_cur * mask
                        else:
                            noise_pred_text = noise_pred_text + noise_pred_text_cur*mask

                # perform classifier-free guidance
                noise_pred = noise_pred_uncond + guidance_scale * \
                    (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)[
                    'prev_sample']

                # apply gradient guidance
                if use_grad_guidance and t < text_format_dict['guidance_start_step']:
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
                            # print(loss)
                            loss_total += loss
                        loss_total.backward()
                    latents = (
                        latents - latents.grad * text_format_dict['color_guidance_weight']).detach().clone()

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

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50,
                      guidance_scale=7.5, latents=None, text_format_dict={}, use_grad_guidance=False):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(
            prompts, negative_prompts)  # [2, 77, 768]

        if len(text_format_dict) > 0:
            if 'font_styles' in text_format_dict and text_format_dict['font_styles'] is not None:
                text_format_dict['font_styles_embs'] = self.get_text_embeds_list(
                    text_format_dict['font_styles'])  # [2, 77, 768]
            else:
                text_format_dict['font_styles_embs'] = None

        # else:
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents,
                                       num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                                       use_grad_guidance=use_grad_guidance, text_format_dict=text_format_dict)  # [1, 4, 64, 64]

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
        for key in self.attention_maps:
            self.attention_maps[key] = []

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

    def remove_evaluation_hooks(self):
        for hook in self.forward_hooks:
            hook.remove()
        self.attention_maps = None
