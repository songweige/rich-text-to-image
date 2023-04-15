import math
import random
import os
import json
import time
import argparse
import torch
import numpy as np
from torchvision import transforms

from models.region_diffusion import RegionDiffusion
from utils.attention_utils import get_token_maps
from utils.richtext_utils import seed_everything, parse_json, get_region_diffusion_input,\
    get_attention_control_input, get_gradient_guidance_input


import gradio as gr
from PIL import Image, ImageOps


help_text = """
Instructions placeholder.
"""


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RegionDiffusion(device)

    def generate(
        text_input: str,
        negative_text: str,
        height: int,
        width: int,
        seed: int,
        steps: int,
        guidance_weight: float,
        color_guidance_weight: float,
    ):
        run_dir = 'results/'
        os.makedirs(run_dir, exist_ok=True)
        # Load region diffusion model.
        steps = 41 if not steps else steps
        guidance_weight = 8.5 if not guidance_weight else guidance_weight

        # parse json to span attributes
        base_text_prompt, style_text_prompts, footnote_text_prompts, footnote_target_tokens,\
            color_text_prompts, color_names, color_rgbs, size_text_prompts_and_sizes, use_grad_guidance = parse_json(
                json.loads(text_input))

        # create control input for region diffusion
        region_text_prompts, region_target_token_ids, base_tokens = get_region_diffusion_input(
            model, base_text_prompt, style_text_prompts, footnote_text_prompts,
            footnote_target_tokens, color_text_prompts, color_names)

        # create control input for cross attention
        text_format_dict = get_attention_control_input(
            model, base_tokens, size_text_prompts_and_sizes)

        # create control input for region guidance
        text_format_dict, color_target_token_ids = get_gradient_guidance_input(
            model, base_tokens, color_text_prompts, color_rgbs, text_format_dict, color_guidance_weight=color_guidance_weight)

        seed_everything(seed)

        # get token maps from plain text to image generation.
        begin_time = time.time()
        if model.attention_maps is None:
            model.register_evaluation_hooks()
        else:
            model.reset_attention_maps()
        plain_img = model.produce_attn_maps([base_text_prompt], [negative_text],
                                            height=height, width=width, num_inference_steps=steps,
                                            guidance_scale=guidance_weight)
        print('time lapses to get attention maps: %.4f' %
              (time.time()-begin_time))
        color_obj_masks, token_maps = get_token_maps(
            model.attention_maps, run_dir, width//8, height//8, color_target_token_ids, seed, retur_vis=True)
        model.masks, token_maps = get_token_maps(
            model.attention_maps, run_dir, width//8, height//8, region_target_token_ids, seed, base_tokens, retur_vis=True)
        color_obj_masks = [transforms.functional.resize(color_obj_mask, (height, width),
                                                        interpolation=transforms.InterpolationMode.BICUBIC,
                                                        antialias=True)
                           for color_obj_mask in color_obj_masks]
        text_format_dict['color_obj_atten'] = color_obj_masks
        model.remove_evaluation_hooks()

        # generate image from rich text
        begin_time = time.time()
        seed_everything(seed)
        rich_img = model.prompt_to_img(region_text_prompts, [negative_text],
                                       height=height, width=width, num_inference_steps=steps,
                                       guidance_scale=guidance_weight, use_grad_guidance=use_grad_guidance,
                                       text_format_dict=text_format_dict)
        print('time lapses to generate image from rich text: %.4f' %
              (time.time()-begin_time))
        cat_img = np.concatenate([plain_img[0], rich_img[0]], 1)
        return [cat_img, token_maps]

    with gr.Blocks() as demo:
        gr.HTML("""<h1 style="font-weight: 900; margin-bottom: 7px;">Expressive Text-to-Image Generation with Rich Text</h1>
                   <p> Visit our <a href="https://rich-text-to-image.github.io/rich-text-to-json.html">rich-text-to-json interface</a> to generate rich-text JSON input.<p/>""")
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label='Rich-text JSON Input',
                    max_lines=1,
                    placeholder='Example: \'{"ops":[{"insert":"a Gothic "},{"attributes":{"color":"#b26b00"},"insert":"church"},{"insert":" in a the sunset with a beautiful landscape in the background.\n"}]}\'')
                negative_prompt = gr.Textbox(
                    label='Negative Prompt',
                    max_lines=1,
                    placeholder='')
                seed = gr.Slider(label='Seed',
                                 minimum=0,
                                 maximum=100000,
                                 step=1,
                                 value=6)
                color_guidance_weight = gr.Slider(label='Color weight lambda',
                                                  minimum=0,
                                                  maximum=2,
                                                  step=0.1,
                                                  value=0.5)
                with gr.Accordion('Other Parameters', open=False):
                    steps = gr.Slider(label='Number of Steps',
                                      minimum=0,
                                      maximum=500,
                                      step=1,
                                      value=41)
                    guidance_weight = gr.Slider(label='CFG weight',
                                                minimum=0,
                                                maximum=50,
                                                step=0.1,
                                                value=8.5)
                    width = gr.Dropdown(choices=[512, 768, 896],
                                        value=512,
                                        label='Width',
                                        visible=True)
                    height = gr.Dropdown(choices=[512, 768, 896],
                                         value=512,
                                         label='height',
                                         visible=True)

                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        generate_button = gr.Button("Generate")

            with gr.Column():
                result = gr.Image(label='Result')
                token_map = gr.Image(label='TokenMap')

        with gr.Row():
            examples = [
                [
                    '{"ops":[{"insert":"a Gothic "},{"attributes":{"color":"#b26b00"},"insert":"church"},{"insert":" in a the sunset with a beautiful landscape in the background."}]}',
                    '',
                    512,
                    512,
                    6,
                    1,
                ],
                [
                    '{"ops": [{"insert": "A pizza with "}, {"attributes": {"size": "50px"}, "insert": "pineapples"}, {"insert": ", pepperonis, and mushrooms on the top, 4k, photorealistic"}]}',
                    'blurry, art, painting, rendering, drawing, sketch, ugly, duplicate, morbid, mutilated, mutated, deformed, disfigured low quality, worst quality',
                    768,
                    896,
                    6,
                    1,
                ],
                [
                    '{"ops":[{"insert":"a "},{"attributes":{"font":"mirza"},"insert":"beautiful garden"},{"insert":" with a "},{"attributes":{"font":"roboto"},"insert":"snow mountain in the background"},{"insert":""}]}',
                    '',
                    512,
                    512,
                    3,
                    1,
                ],
                [
                    '{"ops":[{"insert":"A close-up 4k dslr photo of a "},{"attributes":{"link":"A cat wearing sunglasses and a bandana around its neck."},"insert":"cat"},{"insert":" riding a scooter. Palm trees in the background."}]}',
                    '',
                    512,
                    512,
                    6,
                    1,
                ],
                [
                    {"ops": [{"insert": "a "}, {"attributes": {"font": "slabo"}, "insert": "night sky filled with stars"}, {
                        "insert": " above a "}, {"attributes": {"font": "roboto"}, "insert": "turbulent sea with giant waves"}, {"insert": "\n"}]},
                    '',
                    512,
                    512,
                    6,
                    1,
                ],
            ]
            gr.Examples(examples=examples,
                        inputs=[
                            text_input,
                            negative_prompt,
                            height,
                            width,
                            seed,
                            color_guidance_weight,
                        ],
                        outputs=[
                            result,
                            token_map,
                        ],
                        fn=generate,
                        # cache_examples=True,
                        examples_per_page=20)

        generate_button.click(
            fn=generate,
            inputs=[
                text_input,
                negative_prompt,
                height,
                width,
                seed,
                steps,
                guidance_weight,
                color_guidance_weight,
            ],
            outputs=[result, token_map],
        )

    demo.queue(concurrency_count=1)
    demo.launch(share=False)


if __name__ == "__main__":
    main()
