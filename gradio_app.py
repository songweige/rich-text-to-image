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
from utils.share_btn import community_icon_html, loading_icon_html, share_js, css


help_text = """
If you are encountering an error or not achieving your desired outcome, here are some potential reasons and recommendations to consider:
1. If you format only a portion of a word rather than the complete word, an error may occur. 
2. If you use font color and get completely corrupted results, you may consider decrease the color weight lambda.
3. Consider using a different seed.
"""


canvas_html = """<iframe id='rich-text-root' style='width:100%' height='360px' src='file=utils/rich-text-to-json-iframe.html' frameborder='0' scrolling='no'></iframe>"""
get_js_data = """
async (text_input, negative_prompt, height, width, seed, steps, num_segments, segment_threshold, inject_interval, guidance_weight, color_guidance_weight, rich_text_input, background_aug) => {
  const richEl = document.getElementById("rich-text-root");
  const data = richEl? richEl.contentDocument.body._data : {};
  return [text_input, negative_prompt, height, width, seed, steps, num_segments, segment_threshold, inject_interval, guidance_weight, color_guidance_weight, JSON.stringify(data), background_aug];
}
"""
set_js_data = """
async (text_input) => {
  const richEl = document.getElementById("rich-text-root");
  const data = text_input ? JSON.parse(text_input) : null;
  if (richEl && data) richEl.contentDocument.body.setQuillContents(data);
}
"""

get_window_url_params = """
async (url_params) => {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    return [url_params];
}
"""


def load_url_params(url_params):
    if 'prompt' in url_params:
        return gr.update(visible=True), url_params
    else:
        return gr.update(visible=False), url_params


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
        num_segments: int,
        segment_threshold: float,
        inject_interval: float,
        guidance_weight: float,
        color_guidance_weight: float,
        rich_text_input: str,
        background_aug: bool,
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
        if (text_input == '' or rich_text_input == ''):
            raise gr.Error("Please enter some text.")
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
        if model.selfattn_maps is None and model.crossattn_maps is None:
            model.remove_tokenmap_hooks()
            model.register_tokenmap_hooks()
        else:
            model.reset_attention_maps()
        plain_img = model.produce_attn_maps([base_text_prompt], [negative_text],
                                            height=height, width=width, num_inference_steps=steps,
                                            guidance_scale=guidance_weight)
        print('time lapses to get attention maps: %.4f' %
              (time.time()-begin_time))
        seed_everything(seed)
        color_obj_masks, segments_vis, token_maps = get_token_maps(model.selfattn_maps, model.crossattn_maps, model.n_maps, run_dir,
                                                                   512//8, 512//8, color_target_token_ids[:-1], seed,
                                                                   base_tokens, segment_threshold=segment_threshold, num_segments=num_segments,
                                                                   return_vis=True)
        seed_everything(seed)
        model.masks, segments_vis, token_maps = get_token_maps(model.selfattn_maps, model.crossattn_maps, model.n_maps, run_dir,
                                                               512//8, 512//8, region_target_token_ids[:-1], seed,
                                                               base_tokens, segment_threshold=segment_threshold, num_segments=num_segments,
                                                               return_vis=True)
        color_obj_masks = [transforms.functional.resize(color_obj_mask, (height, width),
                                                        interpolation=transforms.InterpolationMode.BICUBIC,
                                                        antialias=True)
                           for color_obj_mask in color_obj_masks]
        text_format_dict['color_obj_atten'] = color_obj_masks
        model.remove_tokenmap_hooks()

        # generate image from rich text
        begin_time = time.time()
        seed_everything(seed)
        if background_aug:
            bg_aug_end = 500
        else:
            bg_aug_end = 1000
        rich_img = model.prompt_to_img(region_text_prompts, [negative_text],
                                       height=height, width=width, num_inference_steps=steps,
                                       guidance_scale=guidance_weight, use_guidance=use_grad_guidance,
                                       text_format_dict=text_format_dict, inject_selfattn=inject_interval,
                                       bg_aug_end=bg_aug_end)
        print('time lapses to generate image from rich text: %.4f' %
              (time.time()-begin_time))
        return [plain_img[0], rich_img[0], segments_vis, token_maps]

    with gr.Blocks(css=css) as demo:
        url_params = gr.JSON({}, visible=False, label="URL Params")
        gr.HTML("""<h1 style="font-weight: 900; margin-bottom: 7px;">Expressive Text-to-Image Generation with Rich Text</h1>
                   <p> <a href="https://songweige.github.io/">Songwei Ge</a>, <a href="https://taesung.me/">Taesung Park</a>, <a href="https://www.cs.cmu.edu/~junyanz/">Jun-Yan Zhu</a>, <a href="https://jbhuang0604.github.io/">Jia-Bin Huang</a> <p/> 
                   <p> UMD, Adobe, CMU <p/> 
                   <p> <a href="https://huggingface.co/spaces/songweig/rich-text-to-image?duplicate=true"><img src="https://bit.ly/3gLdBN6" style="display:inline;"alt="Duplicate Space"></a> | <a href="https://rich-text-to-image.github.io">[Website]</a> | <a href="https://github.com/SongweiGe/rich-text-to-image">[Code]</a> | <a href="https://arxiv.org/abs/2304.06720">[Paper]</a><p/>
                   <p> For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings.""")
        with gr.Row():
            with gr.Column():
                rich_text_el = gr.HTML(canvas_html, elem_id="canvas_html")
                rich_text_input = gr.Textbox(value="", visible=False)
                text_input = gr.Textbox(
                    label='Rich-text JSON Input',
                    visible=False,
                    max_lines=1,
                    placeholder='Example: \'{"ops":[{"insert":"a Gothic "},{"attributes":{"color":"#b26b00"},"insert":"church"},{"insert":" in a the sunset with a beautiful landscape in the background.\n"}]}\'',
                    elem_id="text_input"
                )
                negative_prompt = gr.Textbox(
                    label='Negative Prompt',
                    max_lines=1,
                    placeholder='Example: poor quality, blurry, dark, low resolution, low quality, worst quality',
                    elem_id="negative_prompt"
                )
                segment_threshold = gr.Slider(label='Token map threshold',
                                              info='(See less area in token maps? Decrease this. See too much area? Increase this.)',
                                              minimum=0,
                                              maximum=1,
                                              step=0.01,
                                              value=0.25)
                inject_interval = gr.Slider(label='Detail preservation',
                                            info='(To preserve more structure from plain-text generation, increase this. To see more rich-text attributes, decrease this.)',
                                            minimum=0,
                                            maximum=1,
                                            step=0.01,
                                            value=0.)
                color_guidance_weight = gr.Slider(label='Color weight',
                                                  info='(To obtain more precise color, increase this, while too large value may cause artifacts.)',
                                                  minimum=0,
                                                  maximum=2,
                                                  step=0.1,
                                                  value=0.5)
                num_segments = gr.Slider(label='Number of segments',
                                         minimum=2,
                                         maximum=20,
                                         step=1,
                                         value=9)
                seed = gr.Slider(label='Seed',
                                 minimum=0,
                                 maximum=100000,
                                 step=1,
                                 value=6,
                                 elem_id="seed"
                                 )
                background_aug = gr.Checkbox(
                    label='Precise region alignment',
                    info='(For strict region alignment, select this option, but beware of potential artifacts when using with style.)',
                    value=True)
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
                    width = gr.Dropdown(choices=[512],
                                        value=512,
                                        label='Width',
                                        visible=True)
                    height = gr.Dropdown(choices=[512],
                                         value=512,
                                         label='height',
                                         visible=True)

                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        generate_button = gr.Button("Generate")
                        load_params_button = gr.Button(
                            "Load from URL Params", visible=True)
            with gr.Column():
                richtext_result = gr.Image(
                    label='Rich-text', elem_id="rich-text-image")
                richtext_result.style(height=512)
                with gr.Row():
                    plaintext_result = gr.Image(
                        label='Plain-text', elem_id="plain-text-image")
                    segments = gr.Image(label='Segmentation')
                with gr.Row():
                    token_map = gr.Image(label='Token Maps')
                with gr.Row(visible=False) as share_row:
                    with gr.Group(elem_id="share-btn-container"):
                        community_icon = gr.HTML(community_icon_html)
                        loading_icon = gr.HTML(loading_icon_html)
                        share_button = gr.Button(
                            "Share to community", elem_id="share-btn")
                        share_button.click(None, [], [], _js=share_js)
        with gr.Row():
            gr.Markdown(help_text)

        with gr.Row():
            footnote_examples = [
                [
                    '{"ops":[{"insert":"A close-up 4k dslr photo of a "},{"attributes":{"link":"A cat wearing sunglasses and a bandana around its neck."},"insert":"cat"},{"insert":" riding a scooter. Palm trees in the background."}]}',
                    '',
                    5,
                    0.3,
                    0,
                    6,
                    1,
                    None,
                    True
                ],
                [
                    '{"ops":[{"insert":"A "},{"attributes":{"link":"kitchen island with a stove with gas burners and a built-in oven "},"insert":"kitchen island"},{"insert":" next to a "},{"attributes":{"link":"an open refrigerator stocked with fresh produce, dairy products, and beverages. "},"insert":"refrigerator"},{"insert":", by James McDonald and Joarc Architects, home, interior, octane render, deviantart, cinematic, key art, hyperrealism, sun light, sunrays, canon eos c 300, ƒ 1.8, 35 mm, 8k, medium - format print"}]}',
                    '',
                    6,
                    0.5,
                    0,
                    6,
                    1,
                    None,
                    True
                ],
                [
                    '{"ops":[{"insert":"A "},{"attributes":{"link":"Happy Kung fu panda art, elder, asian art, volumetric lighting, dramatic scene, ultra detailed, realism, chinese"},"insert":"panda"},{"insert":" standing on a cliff by a waterfall, wildlife photography, photograph, high quality, wildlife, f 1.8, soft focus, 8k, national geographic, award - winning photograph by nick nichols"}]}',
                    '',
                    4,
                    0.3,
                    0,
                    4,
                    1,
                    None,
                    True
                ],
            ]

            gr.Examples(examples=footnote_examples,
                        label='Footnote examples',
                        inputs=[
                            text_input,
                            negative_prompt,
                            num_segments,
                            segment_threshold,
                            inject_interval,
                            seed,
                            color_guidance_weight,
                            rich_text_input,
                            background_aug,
                        ],
                        outputs=[
                            plaintext_result,
                            richtext_result,
                            segments,
                            token_map,
                        ],
                        fn=generate,
                        # cache_examples=True,
                        examples_per_page=20)
        with gr.Row():
            color_examples = [
                [
                    '{"ops":[{"insert":"a beautifule girl with big eye, skin, and long "},{"attributes":{"color":"#00ffff"},"insert":"hair"},{"insert":", t-shirt, bursting with vivid color, intricate, elegant, highly detailed, photorealistic, digital painting,  artstation, illustration, concept art."}]}',
                    'lowres, had anatomy, bad hands, cropped, worst quality',
                    9,
                    0.25,
                    0.3,
                    6,
                    0.5,
                    None,
                    True
                ],
                [
                    '{"ops":[{"insert":"a beautifule girl with big eye, skin, and long "},{"attributes":{"color":"#eeeeee"},"insert":"hair"},{"insert":", t-shirt, bursting with vivid color, intricate, elegant, highly detailed, photorealistic, digital painting,  artstation, illustration, concept art."}]}',
                    'lowres, had anatomy, bad hands, cropped, worst quality',
                    9,
                    0.25,
                    0.3,
                    6,
                    0.1,
                    None,
                    True
                ],
                [
                    '{"ops":[{"insert":"a Gothic "},{"attributes":{"color":"#FD6C9E"},"insert":"church"},{"insert":" in a the sunset with a beautiful landscape in the background."}]}',
                    '',
                    5,
                    0.3,
                    0.5,
                    6,
                    0.5,
                    None,
                    False
                ],
                [
                    '{"ops":[{"insert":"A mesmerizing sight that captures the beauty of a "},{"attributes":{"color":"#4775fc"},"insert":"rose"},{"insert":" blooming, close up"}]}',
                    '',
                    3,
                    0.3,
                    0,
                    9,
                    1,
                    None,
                    False
                ],
                [
                    '{"ops":[{"insert":"A "},{"attributes":{"color":"#FFD700"},"insert":"marble statue of a wolf\'s head and shoulder"},{"insert":", surrounded by colorful flowers michelangelo, detailed, intricate, full of color, led lighting, trending on artstation, 4 k, hyperrealistic, 3 5 mm, focused, extreme details, unreal engine 5, masterpiece "}]}',
                    '',
                    5,
                    0.3,
                    0,
                    5,
                    0.6,
                    None,
                    False
                ],
            ]
            gr.Examples(examples=color_examples,
                        label='Font color examples',
                        inputs=[
                            text_input,
                            negative_prompt,
                            num_segments,
                            segment_threshold,
                            inject_interval,
                            seed,
                            color_guidance_weight,
                            rich_text_input,
                            background_aug,
                        ],
                        outputs=[
                            plaintext_result,
                            richtext_result,
                            segments,
                            token_map,
                        ],
                        fn=generate,
                        # cache_examples=True,
                        examples_per_page=20)

        with gr.Row():
            style_examples = [
                [
                    '{"ops":[{"insert":"a "},{"attributes":{"font":"mirza"},"insert":"beautiful garden"},{"insert":" with a "},{"attributes":{"font":"roboto"},"insert":"snow mountain in the background"},{"insert":""}]}',
                    '',
                    5,
                    0.3,
                    0.2,
                    3,
                    0.5,
                    None,
                    False
                ],
                [
                    '{"ops":[{"attributes":{"link":"the awe-inspiring sky and ocean in the style of J.M.W. Turner"},"insert":"the awe-inspiring sky and sea"},{"insert":" by "},{"attributes":{"font":"mirza"},"insert":"a coast with flowers and grasses in spring"}]}',
                    'worst quality, dark, poor quality',
                    5,
                    0.3,
                    0,
                    9,
                    0.5,
                    None,
                    False
                ],
                [
                    '{"ops":[{"insert":"a "},{"attributes":{"font":"slabo"},"insert":"night sky filled with stars"},{"insert":" above a "},{"attributes":{"font":"roboto"},"insert":"turbulent sea with giant waves"}]}',
                    '',
                    2,
                    0.4,
                    0,
                    6,
                    0.5,
                    None,
                    False
                ],
            ]
            gr.Examples(examples=style_examples,
                        label='Font style examples',
                        inputs=[
                            text_input,
                            negative_prompt,
                            num_segments,
                            segment_threshold,
                            inject_interval,
                            seed,
                            color_guidance_weight,
                            rich_text_input,
                            background_aug,
                        ],
                        outputs=[
                            plaintext_result,
                            richtext_result,
                            segments,
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
                    5,
                    0.3,
                    0,
                    13,
                    1,
                    None,
                    False
                ],
                [
                    '{"ops": [{"insert": "A pizza with pineapple, "}, {"attributes": {"size": "20px"}, "insert": "pepperoni"}, {"insert": ", and mushroom on the top, 4k, photorealistic"}]}',
                    'blurry, art, painting, rendering, drawing, sketch, ugly, duplicate, morbid, mutilated, mutated, deformed, disfigured low quality, worst quality',
                    5,
                    0.3,
                    0,
                    13,
                    1,
                    None,
                    False
                ],
                [
                    '{"ops": [{"insert": "A pizza with pineapple, pepperoni, and "}, {"attributes": {"size": "70px"}, "insert": "mushroom"}, {"insert": " on the top, 4k, photorealistic"}]}',
                    'blurry, art, painting, rendering, drawing, sketch, ugly, duplicate, morbid, mutilated, mutated, deformed, disfigured low quality, worst quality',
                    5,
                    0.3,
                    0,
                    13,
                    1,
                    None,
                    False
                ],
            ]
            gr.Examples(examples=size_examples,
                        label='Font size examples',
                        inputs=[
                            text_input,
                            negative_prompt,
                            num_segments,
                            segment_threshold,
                            inject_interval,
                            seed,
                            color_guidance_weight,
                            rich_text_input,
                            background_aug,
                        ],
                        outputs=[
                            plaintext_result,
                            richtext_result,
                            segments,
                            token_map,
                        ],
                        fn=generate,
                        # cache_examples=True,
                        examples_per_page=20)
        generate_button.click(fn=lambda: gr.update(visible=False), inputs=None, outputs=share_row, queue=False).then(
            fn=generate,
            inputs=[
                text_input,
                negative_prompt,
                height,
                width,
                seed,
                steps,
                num_segments,
                segment_threshold,
                inject_interval,
                guidance_weight,
                color_guidance_weight,
                rich_text_input,
                background_aug
            ],
            outputs=[plaintext_result, richtext_result, segments, token_map],
            _js=get_js_data
        ).then(
            fn=lambda: gr.update(visible=True), inputs=None, outputs=share_row, queue=False)
        text_input.change(
            fn=None, inputs=[text_input], outputs=None, _js=set_js_data, queue=False)
        # load url param prompt to textinput
        load_params_button.click(fn=lambda x: x['prompt'], inputs=[
                                 url_params], outputs=[text_input], queue=False)
        demo.load(
            fn=load_url_params,
            inputs=[url_params],
            outputs=[load_params_button, url_params],
            _js=get_window_url_params
        )
    demo.queue(concurrency_count=1)
    demo.launch(share=False)


if __name__ == "__main__":
    main()
