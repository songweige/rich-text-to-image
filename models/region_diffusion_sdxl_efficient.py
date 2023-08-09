# Adapted from diffusers.pipelines.stable_diffusion.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.py

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
# from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models import AutoencoderKL

from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl.watermark import StableDiffusionXLWatermarker

### cutomized modules
import collections
from functools import partial
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput

from models.unet_2d_condition import UNet2DConditionModel
from utils.attention_utils import CrossAttentionLayers_XL

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class RegionDiffusionXL(DiffusionPipeline, FromSingleFileMixin):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *Textual-Inversion*: [`loaders.TextualInversionLoaderMixin.load_textual_inversion`]
        - *LoRA*: [`loaders.LoraLoaderMixin.load_lora_weights`]
        - *Ckpt*: [`loaders.FromSingleFileMixin.from_single_file`]

    as well as the following saving methods:
        - *LoRA*: [`loaders.LoraLoaderMixin.save_lora_weights`]

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
        self,
        load_path: str = "stabilityai/stable-diffusion-xl-base-0.9",
        device: str = "cuda",
        force_zeros_for_empty_prompt: bool = True,
    ):
        super().__init__()

        # self.register_modules(
        #     vae=vae,
        #     text_encoder=text_encoder,
        #     text_encoder_2=text_encoder_2,
        #     tokenizer=tokenizer,
        #     tokenizer_2=tokenizer_2,
        #     unet=unet,
        #     scheduler=scheduler,
        # )

        # 1. Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained(load_path, subfolder="vae", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(device)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained(load_path, subfolder='tokenizer')
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(load_path, subfolder='tokenizer_2')
        self.text_encoder = CLIPTextModel.from_pretrained(load_path, subfolder='text_encoder', torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(device)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(load_path, subfolder='text_encoder_2', torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(device)

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(load_path, subfolder="unet", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(device)
    
        # 4. Scheduler.
        self.scheduler = EulerDiscreteScheduler.from_pretrained(load_path, subfolder="scheduler")

        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.default_sample_size = self.unet.config.sample_size

        self.watermark = StableDiffusionXLWatermarker()

        self.device_type = device

        self.masks = []
        self.attention_maps = None
        self.color_loss = torch.nn.functional.mse_loss
        self.forward_replacement_hooks = []

    # Overwriting the method from diffusers.pipelines.diffusion_pipeline.DiffusionPipeline
    @property
    def device(self) -> torch.device:
        r"""
        Returns:
            `torch.device`: The torch device on which the pipeline is located.
        """

        return torch.device(self.device_type)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.

        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError("`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.text_encoder_2, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        model_sequence = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )
        model_sequence.extend([self.unet, self.vae])

        hook = None
        for cpu_offloaded_model in model_sequence:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def encode_prompt(
        self,
        prompt,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
            batch_size_neg = len(negative_prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(
                    text_input_ids.to(device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]

                bs_embed, seq_len, _ = prompt_embeds.shape
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            # elif batch_size != len(negative_prompt):
            #     raise ValueError(
            #         f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
            #         f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
            #         " the batch size of `prompt`."
            #     )
            else:
                uncond_tokens = negative_prompt

            negative_prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                # textual inversion: procecss multi-vector tokens if necessary
                if isinstance(self, TextualInversionLoaderMixin):
                    uncond_tokens = self.maybe_convert_prompt(uncond_tokens, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                if do_classifier_free_guidance:
                    # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                    seq_len = negative_prompt_embeds.shape[1]

                    negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder.dtype, device=device)

                    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                    negative_prompt_embeds = negative_prompt_embeds.view(
                        batch_size_neg * num_images_per_prompt, seq_len, -1
                    )

                    # For classifier free guidance, we need to do two forward passes.
                    # Here we concatenate the unconditional and text embeddings into a single batch
                    # to avoid doing two forward passes

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        bs_embed = pooled_prompt_embeds.shape[0]
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        bs_embed = negative_pooled_prompt_embeds.shape[0]
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    @torch.no_grad()
    def sample(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        # Rich-Text args
        use_guidance: bool = False,
        inject_selfattn: float = 0.0,
        inject_background: float = 0.0,
        text_format_dict: Optional[dict] = None,
        run_rich_text: bool = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                TODO
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                TODO
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                TODO

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple. When returning a tuple, the first element is a list with the generated images, and the second
            element is a list of `bool`s denoting whether the corresponding generated image likely represents
            "not-safe-for-work" (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            # TODO: support batched prompts
            batch_size = 1
            # batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        if run_rich_text:
            if inject_selfattn > 0 or inject_background > 0:
                latents_reference = latents.clone().detach()
            n_styles = prompt_embeds.shape[0]-1
            self.masks = [mask.to(dtype=prompt_embeds.dtype) for mask in self.masks]
            print(n_styles, len(self.masks))
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(self.scheduler.timesteps):
                    # predict the noise residual
                    with torch.no_grad():
                        feat_inject_step = t > (1-inject_selfattn) * 1000
                        background_inject_step = i < inject_background * len(self.scheduler.timesteps)
                        latent_model_input = self.scheduler.scale_model_input(latents, t)
                        # import ipdb;ipdb.set_trace()
                        # unconditional prediction
                        noise_pred_uncond_cur = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds[:1],
                                                            cross_attention_kwargs=cross_attention_kwargs,
                                                            added_cond_kwargs={"text_embeds": add_text_embeds[:1], "time_ids": add_time_ids[:1]}
                                                            )['sample']
                        # tokens without any style or footnote
                        self.register_fontsize_hooks(text_format_dict)
                        noise_pred_text_cur = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds[-1:],
                                                            cross_attention_kwargs=cross_attention_kwargs,
                                                            added_cond_kwargs={"text_embeds": add_text_embeds[-1:], "time_ids": add_time_ids[:1]}
                                                            )['sample']
                        self.remove_fontsize_hooks()
                        if inject_selfattn > 0 or inject_background > 0:
                            latent_reference_model_input = self.scheduler.scale_model_input(latents_reference, t)
                            noise_pred_uncond_refer = self.unet(latent_reference_model_input, t, encoder_hidden_states=prompt_embeds[:1],
                                                            cross_attention_kwargs=cross_attention_kwargs,
                                                            added_cond_kwargs={"text_embeds": add_text_embeds[:1], "time_ids": add_time_ids[:1]}
                                                            )['sample']
                            self.register_selfattn_hooks(feat_inject_step)
                            noise_pred_text_refer = self.unet(latent_reference_model_input, t, encoder_hidden_states=prompt_embeds[-1:],
                                                            cross_attention_kwargs=cross_attention_kwargs,
                                                            added_cond_kwargs={"text_embeds": add_text_embeds[-1:], "time_ids": add_time_ids[:1]}
                                                            )['sample']
                            self.remove_selfattn_hooks()
                        noise_pred_uncond = noise_pred_uncond_cur * self.masks[-1]
                        noise_pred_text = noise_pred_text_cur * self.masks[-1]
                        # tokens with style or footnote
                        for style_i, mask in enumerate(self.masks[:-1]):
                            self.register_replacement_hooks(feat_inject_step)
                            noise_pred_text_cur = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds[style_i+1:style_i+2],
                                                            cross_attention_kwargs=cross_attention_kwargs,
                                                            added_cond_kwargs={"text_embeds": add_text_embeds[style_i+1:style_i+2], "time_ids": add_time_ids[:1]}
                                                            )['sample']
                            self.remove_replacement_hooks()
                            noise_pred_uncond = noise_pred_uncond + noise_pred_uncond_cur*mask
                            noise_pred_text = noise_pred_text + noise_pred_text_cur*mask

                    # perform guidance
                    noise_pred = noise_pred_uncond + guidance_scale * \
                        (noise_pred_text - noise_pred_uncond)

                    if do_classifier_free_guidance and guidance_rescale > 0.0:
                        # TODO: Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        # noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
                        raise NotImplementedError

                    if inject_selfattn > 0 or background_inject_step > 0:
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
                        self.unet.to(device='cpu')
                        torch.cuda.empty_cache()
                        with torch.enable_grad():
                            if not latents.requires_grad:
                                latents.requires_grad = True
                            latents_0 = self.predict_x0(latents, noise_pred, t).to(dtype=torch.bfloat16)
                            # latents_0 = self.predict_x0(latents, noise_pred, t).to(dtype=latents.dtype)
                            latents_inp = latents_0 / self.vae.config.scaling_factor
                            self.vae.to(dtype=latents_inp.dtype)
                            # import ipdb;ipdb.set_trace()
                            imgs = self.vae.decode(latents_inp).sample
                            # imgs = self.vae.decode(latents_inp.float()).sample
                            imgs = (imgs / 2 + 0.5).clamp(0, 1)
                            loss_total = 0.
                            for attn_map, rgb_val in zip(text_format_dict['color_obj_atten'], text_format_dict['target_RGB']):
                                avg_rgb = (
                                    imgs*attn_map[:, 0]).sum(2).sum(2)/attn_map[:, 0].sum()
                                loss = self.color_loss(
                                    avg_rgb, rgb_val[:, :, 0, 0])*100
                                loss_total += loss
                            # import ipdb;ipdb.set_trace()
                            loss_total.backward()
                        latents = (
                            latents - latents.grad * text_format_dict['color_guidance_weight'] * text_format_dict['color_obj_atten_all']).detach().clone().to(dtype=prompt_embeds.dtype)
                        self.unet.to(device=latents.device)

                    # apply background injection
                    if i == int(inject_background * len(self.scheduler.timesteps)) and inject_background > 0:
                        latents = latents_reference * self.masks[-1] + latents * \
                            (1-self.masks[-1])

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)
        else:
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if do_classifier_free_guidance and guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

        # make sure the VAE is in float32 mode, as it overflows in float16
        self.vae.to(dtype=torch.float32)

        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(latents.dtype)
            self.vae.decoder.conv_in.to(latents.dtype)
            self.vae.decoder.mid_block.to(latents.dtype)
        else:
            latents = latents.float()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
            return StableDiffusionXLPipelineOutput(images=image)

        image = self.watermark.apply_watermark(image)
        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

    def predict_x0(self, x_t, eps_t, t):
        alpha_t = self.scheduler.alphas_cumprod[t.cpu().long().item()]
        return (x_t - eps_t * torch.sqrt(1-alpha_t)) / torch.sqrt(alpha_t)

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
                if name in CrossAttentionLayers_XL and n_maps[name] > 10:
                # if n_maps[name] > 10:
                    if name in crossattn_maps:
                        crossattn_maps[name] += out[1][0].detach().cpu()[1:2]
                    else:
                        crossattn_maps[name] = out[1][0].detach().cpu()[1:2]
                # For visualization
                # crossattn_maps[name].append(out[1][0].detach().cpu()[1:2])
            else:
                assert out[1][0].shape[-1] != 77
                # if name in SelfAttentionLayers and n_maps[name] > 10:
                if n_maps[name] > 10:
                    if name in selfattn_maps:
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
        
    def register_replacement_hooks(self, feat_inject_step=False):
        r"""Function for registering hooks to replace self attention.
        """
        self.forward_replacement_hooks = []

        def replace_activations(name, module, args):
            r"""
            PyTorch Forward hook to save outputs at each forward pass.
            """
            if 'attn1' in name:
                modified_args = (args[0], self.self_attention_maps_cur[name].to(args[0].device))
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
                             self.self_attention_maps_cur[name].to(args[0].device))
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

    def remove_replacement_hooks(self):
        for hook in self.forward_replacement_hooks:
            hook.remove()


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
                activations[name] = out[1][1].detach().cpu()

        def save_resnet_activations(activations, name, module, inp, out):
            r"""
            PyTorch Forward hook to save outputs at each forward pass.
            """
            # out[0] - final output of residual layer
            # out[1] - residual hidden feature
            # import ipdb;ipdb.set_trace()
            assert out[1].shape[-1] == 64
            activations[name] = out[1].detach().cpu()
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

        if text_format_dict['word_pos'] is not None and text_format_dict['font_size'] is not None:
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