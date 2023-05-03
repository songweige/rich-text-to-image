import os
import json
import time
import argparse
import imageio
import torch
import numpy as np
from torchvision import transforms

from models.region_diffusion import RegionDiffusion
from utils.attention_utils import get_token_maps
from utils.richtext_utils import seed_everything, parse_json, get_region_diffusion_input,\
    get_attention_control_input, get_gradient_guidance_input


def main(args, param):

    # Create the folder to store outputs.
    run_dir = args.run_dir
    os.makedirs(args.run_dir, exist_ok=True)

    # Load region diffusion model.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RegionDiffusion(device)

    # parse json to span attributes
    base_text_prompt, style_text_prompts, footnote_text_prompts, footnote_target_tokens,\
        color_text_prompts, color_names, color_rgbs, size_text_prompts_and_sizes, use_grad_guidance = parse_json(
            param['text_input'])

    # create control input for region diffusion
    region_text_prompts, region_target_token_ids, base_tokens = get_region_diffusion_input(
        model, base_text_prompt, style_text_prompts, footnote_text_prompts,
        footnote_target_tokens, color_text_prompts, color_names)

    # create control input for cross attention
    text_format_dict = get_attention_control_input(
        model, base_tokens, size_text_prompts_and_sizes)

    # create control input for region guidance
    text_format_dict, color_target_token_ids = get_gradient_guidance_input(
        model, base_tokens, color_text_prompts, color_rgbs, text_format_dict, color_guidance_weight=args.color_guidance_weight)

    height = param['height']
    width = param['width']
    seed = param['noise_index']
    negative_text = param['negative_prompt']
    seed_everything(seed)

    # get token maps from plain text to image generation.
    begin_time = time.time()
    if model.attention_maps is None:
        model.register_tokenmap_hooks()
    else:
        model.reset_attention_maps()
    plain_img = model.produce_attn_maps([base_text_prompt], [negative_text],
                                        height=height, width=width, num_inference_steps=param['steps'],
                                        guidance_scale=param['guidance_weight'])
    fn_base = os.path.join(run_dir, 'seed%d_plain.png' % (seed))
    imageio.imwrite(fn_base, plain_img[0])
    print('time lapses to get attention maps: %.4f' % (time.time()-begin_time))
    seed_everything(seed)
    color_obj_masks = get_token_maps(model.selfattn_maps, model.crossattn_maps, model.n_maps, run_dir,
                                     height//8, width//8, color_target_token_ids[:-1], seed,
                                     base_tokens, segment_threshold=args.segment_threshold, num_segments=args.num_segments)
    color_obj_masks = [transforms.functional.resize(color_obj_mask, (height, width),
                                                    interpolation=transforms.InterpolationMode.BICUBIC,
                                                    antialias=True)
                       for color_obj_mask in color_obj_masks]
    text_format_dict['color_obj_atten'] = color_obj_masks
    seed_everything(seed)
    model.masks = get_token_maps(model.selfattn_maps, model.crossattn_maps, model.n_maps, run_dir,
                                 height//8, width//8, region_target_token_ids[:-1], seed,
                                 base_tokens, segment_threshold=args.segment_threshold, num_segments=args.num_segments)
    model.remove_tokenmap_hooks()

    # generate image from rich text
    begin_time = time.time()
    seed_everything(seed)
    rich_img = model.prompt_to_img(region_text_prompts, [negative_text],
                                   height=height, width=width, num_inference_steps=param['steps'],
                                   guidance_scale=param['guidance_weight'], use_guidance=use_grad_guidance,
                                   inject_selfattn=args.inject_selfattn, bg_aug_end=args.bg_aug_end,
                                   text_format_dict=text_format_dict)
    print('time lapses to generate image from rich text: %.4f' %
          (time.time()-begin_time))
    fn_style = os.path.join(run_dir, 'seed%d_rich.png' % (seed))
    imageio.imwrite(fn_style, rich_img[0])
    # imageio.imwrite(fn_cat, np.concatenate([img[0], rich_img[0]], 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, default='results/')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--seed', type=int, default=6)
    parser.add_argument('--sample_steps', type=int, default=41)
    parser.add_argument('--rich_text_json', type=str,
                        default='{"ops":[{"insert":"A close-up 4k dslr photo of a "},{"attributes":{"link":"A cat wearing sunglasses and a bandana around its neck."},"insert":"cat"},{"insert":" riding a scooter. There are palm trees in the background."}]}')
    parser.add_argument('--negative_prompt', type=str, default='')
    parser.add_argument('--guidance_weight', type=float, default=8.5)
    parser.add_argument('--color_guidance_weight', type=float, default=0.5)
    parser.add_argument('--inject_selfattn', type=float, default=0.)
    parser.add_argument('--segment_threshold', type=float, default=0.3)
    parser.add_argument('--num_segments', type=int, default=9)
    parser.add_argument('--bg_aug_end', type=int, default=1000)
    args = parser.parse_args()
    param = {
        'text_input': json.loads(args.rich_text_json),
        'height': args.height,
        'width': args.width,
        'guidance_weight': args.guidance_weight,
        'steps': args.sample_steps,
        'noise_index': args.seed,
        'negative_prompt': args.negative_prompt,
    }

    main(args, param)
