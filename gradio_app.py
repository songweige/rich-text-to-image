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
If you are encountering an error or not achieving your desired outcome, here are some potential reasons and recommendations to consider:
1. If you format only a portion of a word rather than the complete word, an error may occur. 
2. The token map may not always accurately capture the region of the formatted tokens. If you're experiencing this problem, experiment with selecting more or fewer tokens to expand or reduce the area covered by the token maps.
3. If you use font color and get completely corrupted results, you may consider decrease the color weight lambda.
4. Consider using a different seed.
"""


canvas_html = """<iframe id='rich-text-root' style='width:100%' height='360px' src='file=rich-text-to-json-iframe.html' frameborder='0' scrolling='no'></iframe>"""
get_js_data = """
async (text_input, negative_prompt, height, width, seed, steps, guidance_weight, color_guidance_weight, rich_text_input) => {
  const richEl = document.getElementById("rich-text-root");
  const data = richEl? richEl.contentDocument.body._data : {};
  return [text_input, negative_prompt, height, width, seed, steps, guidance_weight, color_guidance_weight, JSON.stringify(data)];
}
"""
set_js_data = """
async (text_input) => {
  const richEl = document.getElementById("rich-text-root");
  const data = text_input ? JSON.parse(text_input) : null;
  if (richEl && data) richEl.contentDocument.body.setQuillContents(data);
}
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
        rich_text_input: str
    ):
        run_dir = 'results/'
        os.makedirs(run_dir, exist_ok=True)
        # Load region diffusion model.
        height = int(height)
        width = int(width)
        steps = 41 if not steps else steps
        guidance_weight = 8.5 if not guidance_weight else guidance_weight
        text_input = rich_text_input if rich_text_input != '' else text_input
        print('text_input', text_input)
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
        return [plain_img[0], rich_img[0], token_maps]

    with gr.Blocks() as demo:
        gr.HTML("""<h1 style="font-weight: 900; margin-bottom: 7px;">Expressive Text-to-Image Generation with Rich Text</h1>
                   <p> <a href="https://rich-text-to-image.github.io">[Website]</a> | <a href="https://github.com/SongweiGe/rich-text-to-image">[Code]</a> | <a href="https://arxiv.org/abs/2304.06720">[Paper]</a> <p/> """)
        with gr.Row():
            with gr.Column():
                rich_text_el = gr.HTML(canvas_html, elem_id="canvas_html")
                rich_text_input = gr.Textbox(value="", visible=False)
                text_input = gr.Textbox(
                    label='Rich-text JSON Input',
                    visible=False,
                    max_lines=1,
                    placeholder='Example: \'{"ops":[{"insert":"a Gothic "},{"attributes":{"color":"#b26b00"},"insert":"church"},{"insert":" in a the sunset with a beautiful landscape in the background.\n"}]}\'')
                negative_prompt = gr.Textbox(
                    label='Negative Prompt',
                    max_lines=1,
                    placeholder='Example: poor quality, blurry, dark, low resolution, low quality, worst quality')
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
                    width = gr.Dropdown(choices=[512, 768],
                                    value=512,
                                    label='Width',
                                    visible=True)
                    height = gr.Dropdown(choices=[512, 768],
                                    value=512,
                                    label='height',
                                    visible=True)
                    
                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        generate_button = gr.Button("Generate")

            with gr.Column():
                richtext_result = gr.Image(label='Rich-text')
                richtext_result.style(height=512)
                with gr.Row():
                    plaintext_result = gr.Image(label='Plain-text')
                    token_map = gr.Image(label='Token Maps')

        with gr.Row():
            gr.Markdown(help_text)

        with gr.Row():
            style_examples = [
                [
                    '{"ops":[{"insert":"a "},{"attributes":{"font":"slabo"},"insert":"night sky filled with stars"},{"insert":" above a "},{"attributes":{"font":"roboto"},"insert":"turbulent sea with giant waves"}]}',
                    '',
                    512,
                    512,
                    6,
                    1,
                    None
                ],
                [
                    '{"ops":[{"insert":"a "},{"attributes":{"font":"mirza"},"insert":"beautiful garden"},{"insert":" with a "},{"attributes":{"font":"roboto"},"insert":"snow mountain in the background"},{"insert":""}]}',
                    '',
                    512,
                    512,
                    3,
                    1,
                    None
                ],
                [
                    '{"ops":[{"attributes":{"link":"the awe-inspiring sky and ocean in the style of J.M.W. Turner"},"insert":"the awe-inspiring sky and sea"},{"insert":" by "},{"attributes":{"font":"mirza"},"insert":"a coast with flowers and grasses in spring"}]}',
                    'worst quality, dark, poor quality',
                    512,
                    512,
                    9,
                    1,
                    None
                ],
            ]
            gr.Examples(examples=style_examples,
                        label='Font style examples',
                        inputs=[
                            text_input,
                            negative_prompt,
                            height,
                            width,
                            seed,
                            color_guidance_weight,
                            rich_text_input,
                        ],
                        outputs=[
                            plaintext_result,
                            richtext_result,
                            token_map,
                        ],
                        fn=generate,
                        # cache_examples=True,
                        examples_per_page=20)
        with gr.Row():
            footnote_examples = [
                [
                    '{"ops":[{"insert":"A close-up 4k dslr photo of a "},{"attributes":{"link":"A cat wearing sunglasses and a bandana around its neck."},"insert":"cat"},{"insert":" riding a scooter. Palm trees in the background."}]}',
                    '',
                    512,
                    512,
                    6,
                    1,
                    None
                ],
                [
                    '{"ops":[{"insert":"A "},{"attributes":{"link":"kitchen island with a built-in oven and a stove with gas burners "},"insert":"kitchen island"},{"insert":" next to a "},{"attributes":{"link":"an open refrigerator stocked with fresh produce, dairy products, and beverages. "},"insert":"refrigerator"},{"insert":", by James McDonald and Joarc Architects, home, interior, octane render, deviantart, cinematic, key art, hyperrealism, sun light, sunrays, canon eos c 300, ƒ 1.8, 35 mm, 8k, medium - format print"}]}',
                    '',
                    512,
                    512,
                    6,
                    1,
                    None
                ],
                [
                    '{"ops":[{"insert":"A "},{"attributes":{"link":"Art inspired by kung fu panda, elder, asian art, volumetric lighting, dramatic scene, ultra detailed, realism, chinese"},"insert":"panda"},{"insert":" standing on a cliff by a waterfall, wildlife photography, photograph, high quality, wildlife, f 1.8, soft focus, 8k, national geographic, award - winning photograph by nick nichols"}]}',
                    '',
                    512,
                    512,
                    6,
                    1,
                    None
                ],
            ]
            
            gr.Examples(examples=footnote_examples,
                        label='Footnote examples',
                        inputs=[
                            text_input,
                            negative_prompt,
                            height,
                            width,
                            seed,
                            color_guidance_weight,
                            rich_text_input,
                        ],
                        outputs=[
                            plaintext_result,
                            richtext_result,
                            token_map,
                        ],
                        fn=generate,
                        # cache_examples=True,
                        examples_per_page=20)
        with gr.Row():
            color_examples = [
                [
                    '{"ops":[{"insert":"a Gothic "},{"attributes":{"color":"#b26b00"},"insert":"church"},{"insert":" in a the sunset with a beautiful landscape in the background."}]}',
                    '',
                    512,
                    512,
                    6,
                    1,
                    None
                ],
                [
                    '{"ops":[{"insert":"A mesmerizing sight that captures the beauty of a "},{"attributes":{"color":"#4775fc"},"insert":"rose"},{"insert":" blooming, close up"}]}',
                    '',
                    512,
                    512,
                    9,
                    1,
                    None
                ],
                [
                    '{"ops":[{"insert":"A "},{"attributes":{"color":"#FFD700"},"insert":"marble statue of a wolf\'s head and shoulder"},{"insert":", surrounded by colorful flowers michelangelo, detailed, intricate, full of color, led lighting, trending on artstation, 4 k, hyperrealistic, 3 5 mm, focused, extreme details, unreal engine 5, masterpiece "}]}',
                    '',
                    512,
                    512,
                    5,
                    0.6,
                    None
                ],
            ]
            gr.Examples(examples=color_examples,
                        label='Font color examples',
                        inputs=[
                            text_input,
                            negative_prompt,
                            height,
                            width,
                            seed,
                            color_guidance_weight,
                            rich_text_input,
                        ],
                        outputs=[
                            plaintext_result,
                            richtext_result,
                            token_map,
                        ],
                        fn=generate,
                        # cache_examples=True,
                        examples_per_page=20)
        with gr.Row():
            size_examples = [
                [
                    '{"ops": [{"insert": "A pizza with "}, {"attributes": {"size": "60px"}, "insert": "pineapple"}, {"insert": ", pepperoni, and mushroom on the top, 4k, photorealistic"}]}',
                    'blurry, art, painting, rendering, drawing, sketch, ugly, duplicate, morbid, mutilated, mutated, deformed, disfigured low quality, worst quality',
                    512,
                    512,
                    13,
                    1,
                    None
                ],
                [
                    '{"ops": [{"insert": "A pizza with pineapple, "}, {"attributes": {"size": "20px"}, "insert": "pepperoni"}, {"insert": ", and mushroom on the top, 4k, photorealistic"}]}',
                    'blurry, art, painting, rendering, drawing, sketch, ugly, duplicate, morbid, mutilated, mutated, deformed, disfigured low quality, worst quality',
                    512,
                    512,
                    13,
                    1,
                    None
                ],
                [
                    '{"ops": [{"insert": "A pizza with pineapple, pepperoni, and "}, {"attributes": {"size": "70px"}, "insert": "mushroom"}, {"insert": " on the top, 4k, photorealistic"}]}',
                    'blurry, art, painting, rendering, drawing, sketch, ugly, duplicate, morbid, mutilated, mutated, deformed, disfigured low quality, worst quality',
                    512,
                    512,
                    13,
                    1,
                    None
                ],
            ]
            gr.Examples(examples=size_examples,
                        label='Font size examples',
                        inputs=[
                            text_input,
                            negative_prompt,
                            height,
                            width,
                            seed,
                            color_guidance_weight,
                            rich_text_input,
                        ],
                        outputs=[
                            plaintext_result,
                            richtext_result,
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
                rich_text_input
            ],
            outputs=[plaintext_result, richtext_result, token_map],
            _js=get_js_data
        )
        text_input.change(fn=None, inputs=[text_input], outputs=None, _js=set_js_data, queue=False)
    demo.queue(concurrency_count=1)
    demo.launch(share=False)


if __name__ == "__main__":
    main()