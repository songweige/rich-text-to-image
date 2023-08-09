import sys
sys.path.append('.')

import torch
import argparse
import numpy as np
import os
import imageio
import torchvision
from diffusers import StableDiffusionPipeline
from models.region_diffusion import RegionDiffusion
from utils import ptp_utils
from utils.richtext_utils import find_nearest_color, seed_everything
from utils.attention_utils import get_token_maps


COLORS_common = {
    'brown': [165, 42, 42],
    'red': [255, 0, 0],
    'pink': [253, 108, 158],
    'orange': [255, 165, 0],
    'yellow': [255, 255, 0],
    'purple': [128, 0, 128],
    'green': [0, 128, 0],
    'blue': [0, 0, 255],
    'white': [255, 255, 255],
    'gray': [128, 128, 128],
    'black': [0, 0, 0],
    'crimson': [220, 20, 60],
    'maroon': [128, 0, 0],
    'cyan': [0, 255, 255],
    'azure': [240, 255, 255],
    'turquoise': [64, 224, 208],
    'magenta': [255, 0, 255],
}

COLORS_html = {
    'Fire Brick red': [178, 34, 34],
    'Salmon red': [250, 128, 114],
    'Coral orange': [255, 127, 80],
    'Tomato orange': [255, 99, 71],
    'Peach Puff orange': [255, 218, 185],
    'Moccasin orange': [255, 228, 181],
    'Goldenrod yellow': [218, 165, 32],
    'Olive yellow': [128, 128, 0],
    'Gold yellow': [255, 215, 0],
    'Lavender purple': [230, 230, 250],
    'Indigo purple': [75, 0, 130],
    'Thistle purple': [216, 191, 216],
    'Plum purple': [221, 160, 221],
    'Violet purple': [238, 130, 238],
    'Orchid purple': [218, 112, 214],
    'Chartreuse green': [127, 255, 0],
    'Lawn green': [124, 252, 0],
    'Lime green': [50, 205, 50],
    'Forest green': [34, 139, 34],
    'Spring green': [0, 255, 127],
    'Sea green': [46, 139, 87],
    'Sky blue': [135, 206, 235],
    'Dodger blue': [30, 144, 255],
    'Steel blue': [70, 130, 180],
    'Navy blue': [0, 0, 128],
    'Slate blue': [106, 90, 205],
    'Wheat brown': [245, 222, 179],
    'Tan brown': [210, 180, 140],
    'Peru brown': [205, 133, 63],
    'Chocolate brown': [210, 105, 30],
    'Sienna brown': [160, 82, 4],
    'Floral White': [255, 250, 240],
    'Honeydew White': [240, 255, 240],
}

COLORS_rgb = {
    'color of RGB values [68, 17, 237]': [68, 17, 237],
    'color of RGB values [173, 99, 227]': [173, 99, 227],
    'color of RGB values [48, 131, 172]': [48, 131, 172],
    'color of RGB values [198, 234, 45]': [198, 234, 45],
    'color of RGB values [182, 53, 74]': [182, 53, 74],
    'color of RGB values [29, 139, 118]': [29, 139, 118],
    'color of RGB values [105, 96, 172]': [105, 96, 172],
    'color of RGB values [216, 118, 105]': [216, 118, 105],
    'color of RGB values [88, 119, 37]': [88, 119, 37],
    'color of RGB values [189, 132, 98]': [189, 132, 98],
    'color of RGB values [78, 174, 11]': [78, 174, 11],
    'color of RGB values [39, 126, 109]': [39, 126, 109],
    'color of RGB values [236, 81, 34]': [236, 81, 34],
    'color of RGB values [157, 69, 64]': [157, 69, 64],
    'color of RGB values [67, 192, 60]': [67, 192, 60],
    'color of RGB values [181, 57, 181]': [181, 57, 181],
    'color of RGB values [71, 240, 139]': [71, 240, 139],
    'color of RGB values [34, 153, 226]': [34, 153, 226],
    'color of RGB values [47, 221, 120]': [47, 221, 120],
    'color of RGB values [219, 100, 27]': [219, 100, 27],
    'color of RGB values [228, 168, 120]': [228, 168, 120],
    'color of RGB values [195, 31, 8]': [195, 31, 8],
    'color of RGB values [84, 142, 64]': [84, 142, 64],
    'color of RGB values [104, 120, 31]': [104, 120, 31],
    'color of RGB values [240, 209, 78]': [240, 209, 78],
    'color of RGB values [38, 175, 96]': [38, 175, 96],
    'color of RGB values [116, 233, 180]': [116, 233, 180],
    'color of RGB values [205, 196, 126]': [205, 196, 126],
    'color of RGB values [56, 107, 26]': [56, 107, 26],
    'color of RGB values [200, 55, 100]': [200, 55, 100],
    'color of RGB values [35, 21, 185]': [35, 21, 185],
    'color of RGB values [77, 26, 73]': [77, 26, 73],
    'color of RGB values [216, 185, 14]': [216, 185, 14],
    'color of RGB values [53, 21, 50]': [53, 21, 50],
    'color of RGB values [222, 80, 195]': [222, 80, 195],
    'color of RGB values [103, 168, 84]': [103, 168, 84],
    'color of RGB values [57, 51, 218]': [57, 51, 218],
    'color of RGB values [143, 77, 162]': [143, 77, 162],
    'color of RGB values [25, 75, 226]': [25, 75, 226],
    'color of RGB values [99, 219, 32]': [99, 219, 32],
    'color of RGB values [211, 22, 52]': [211, 22, 52],
    'color of RGB values [162, 239, 198]': [162, 239, 198],
    'color of RGB values [40, 226, 144]': [40, 226, 144],
    'color of RGB values [208, 211, 9]': [208, 211, 9],
    'color of RGB values [231, 121, 82]': [231, 121, 82],
    'color of RGB values [108, 105, 52]': [108, 105, 52],
    'color of RGB values [105, 28, 226]': [105, 28, 226],
    'color of RGB values [31, 94, 190]': [31, 94, 190],
    'color of RGB values [116, 6, 93]': [116, 6, 93],
    'color of RGB values [61, 82, 239]': [61, 82, 239],
}

OBJECTS = [
    'shirt',
    'pants',
    'car',
    'fruit',
    'vegetable',
    'flower',
    'bottle beverage',
    'plant',
    'candy',
    'toy',
    'gem',
    'church',
]


BASE_PROMPTS = [
    'a man wearing a shirt',
    'a woman wearing pants',
    'a car in the street',
    'a basket of fruit',
    'a bowl of vegetable',
    'a flower in a vase',
    'a bottle of beverage on the table',
    'a plant in the garden',
    'a candy on the table',
    'a toy on the floor',
    'a gem on the ground',
    'a church with beautiful landscape in the background',
]


NUM_DIFFUSION_STEPS = 41
GUIDANCE_SCALE = 8.5


def main(args):
    negative_text = ''
    text_format_dict = {
        'guidance_start_step': 999,
        'color_guidance_weight': 1
    }
    height = 512
    width = 512
    init_seed = args.seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    region_model = RegionDiffusion(device)
    ldm_stable = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5").to(device)
    tokenizer = ldm_stable.tokenizer

    ours_min_dis = []
    ours_avg_dis = []
    p2p_min_dis = []
    p2p_avg_dis = []
    plain_min_dis = []
    plain_avg_dis = []

    if args.category == 'common':
        COLORS = COLORS_common
    elif args.category == 'html':
        COLORS = COLORS_html
    elif args.category == 'rgb':
        COLORS = COLORS_rgb

    for seed in range(init_seed, init_seed+3):
        seed_everything(seed)
        latent = torch.randn((1, 4, height // 8, width // 8), device='cuda')

        for text_prompt, object_name in zip(BASE_PROMPTS, OBJECTS):
            base_name = object_name
            region_model.register_tokenmap_hooks()
            seed_everything(seed)
            img_base = region_model.produce_attn_maps([text_prompt], [negative_text],
                                                      height=height, width=width, num_inference_steps=NUM_DIFFUSION_STEPS,
                                                      guidance_scale=GUIDANCE_SCALE, latents=latent)
            # create control input for region diffusion
            obj_token_ids = []
            base_tokens = region_model.tokenizer._tokenize(text_prompt)
            obj_token_ids.append([])
            color_tokens = region_model.tokenizer._tokenize(object_name)
            for color_token in color_tokens:
                obj_token_ids[-1].append(base_tokens.index(color_token)+1)
            obj_token_ids_all = [id for ids in obj_token_ids for id in ids]
            obj_token_ids_rest = [id for id in range(
                1, len(base_tokens)+1) if id not in obj_token_ids_all]
            obj_token_ids.append(obj_token_ids_rest)
            obj_token_ids = [torch.LongTensor(obj_token_id)
                             for obj_token_id in obj_token_ids]
            region_model.masks = get_token_maps(region_model.selfattn_maps, region_model.crossattn_maps, region_model.n_maps, save_path,
                                     height//8, width//8, obj_token_ids[:-1], seed,
                                     base_tokens, segment_threshold=0.3, num_segments=15)
            region_masks = [torchvision.transforms.functional.resize(region_mask, (height, width),
                                                                     interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True).cpu().clamp(0, 1).numpy()
                            for region_mask in region_model.masks]
            color_obj_atten_all = torch.zeros_like(region_model.masks[-1])
            for obj_mask in region_model.masks[:-1]:
                color_obj_atten_all += obj_mask
            color_obj_masks = [torchvision.transforms.functional.resize(color_obj_mask, (height, width),
                                                                        interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)
                               for color_obj_mask in region_model.masks]
            text_format_dict['color_obj_atten_all'] = color_obj_atten_all
            text_format_dict['color_obj_atten'] = color_obj_masks
            if args.save_img:
                imageio.imwrite(os.path.join(
                    save_path, 'plain_%s_seed%d.png' % (base_name, seed)), img_base[0])
            region_model.remove_tokenmap_hooks()
            images_ours = []
            images_p2p = []
            for color_name in COLORS:
                rgb = torch.FloatTensor(COLORS[color_name])[
                    None, :, None, None]/255.
                text_format_dict['target_RGB'] = [rgb.cuda()]
                p2p_name = os.path.join(
                    save_path, 'p2p_%s_%s_%d.png' % (base_name, color_name, seed))
                ours_name = os.path.join(
                    save_path, 'ours_%s_%s_%d.png' % (base_name, color_name, seed))
                if not args.load_previous:
                    # add base prompt as the other attention maps
                    nearest_color_name = find_nearest_color(rgb)
                    text_prompts_rich = ['%s %s' %
                                         (nearest_color_name, object_name)]
                    text_prompts_rich.append(text_prompt)
                    seed_everything(seed)
                    img_ours = region_model.prompt_to_img(text_prompts_rich, [negative_text],
                                                          height=height, width=width, num_inference_steps=NUM_DIFFUSION_STEPS,
                                                          guidance_scale=GUIDANCE_SCALE, text_format_dict=text_format_dict, latents=latent,
                                                          use_guidance=True, inject_selfattn=0.2,
                                                          inject_background=0.3)
                    if args.save_img:
                        imageio.imwrite(ours_name, img_ours[0])
                    if len(images_ours) == 0:
                        images_ours.append(img_base[0])
                    images_ours.append(img_ours[0])
                    img_ours = img_ours.astype(float)
                    # prompt to prompt
                    text_prompt_p2p = text_prompt.replace(
                        object_name, color_name+' '+object_name)
                    text_prompts_p2p = [text_prompt, text_prompt_p2p]
                    controller = ptp_utils.AttentionRefine(text_prompts_p2p, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
                                                           self_replace_steps=.4, tokenizer=tokenizer)
                    seed_everything(seed)
                    img_p2p, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, text_prompts_p2p, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS,
                                                                   guidance_scale=GUIDANCE_SCALE, generator=None, low_resource=False)

                    img_p2p = img_p2p.astype(float)
                    if args.save_img:
                        imageio.imwrite(os.path.join(
                            save_path, 'p2p_%s.png' % (base_name)), img_p2p[0])
                        imageio.imwrite(p2p_name, img_p2p[1])
                    if len(images_p2p) == 0:
                        images_p2p.append(img_p2p[0])
                    img_p2p = img_p2p[1]
                else:
                    img_ours = imageio.imread(ours_name)
                    img_p2p = imageio.imread(p2p_name)
                plain_background = np.zeros_like(
                    img_ours) if color_name != 'black' else np.ones_like(img_ours)*255
                for region_id, region_mask in enumerate(region_masks[:-1]):
                    # metrics for plain text result
                    composed_region = (region_mask[:, 0, :, :, None]*img_base + (
                        1-region_mask[:, 0, :, :, None])*plain_background).round().astype('uint8')
                    rgb_np = rgb.permute([0, 2, 3, 1]).cpu().numpy()
                    euc_dis = np.sqrt(
                        ((composed_region/255.-rgb_np)**2).sum(-1))
                    plain_cur_min_dis = np.min(euc_dis)
                    euc_dis = np.sqrt(((img_base/255.-rgb_np)**2).sum(-1))
                    plain_cur_avg_dis = (
                        euc_dis * region_mask[:, 0, :, :]).sum() / region_mask[:, 0, :, :].sum()
                    plain_min_dis.append(plain_cur_min_dis)
                    plain_avg_dis.append(plain_cur_avg_dis)
                    # metrics for ours
                    composed_region = (region_mask[:, 0, :, :, None]*img_ours + (
                        1-region_mask[:, 0, :, :, None])*plain_background).round().astype('uint8')
                    rgb_np = rgb.permute([0, 2, 3, 1]).cpu().numpy()
                    euc_dis = np.sqrt(
                        ((composed_region/255.-rgb_np)**2).sum(-1))
                    min_dis = np.min(euc_dis)
                    euc_dis = np.sqrt(((img_ours/255.-rgb_np)**2).sum(-1))
                    avg_dis = (
                        euc_dis * region_mask[:, 0, :, :]).sum() / region_mask[:, 0, :, :].sum()
                    ours_min_dis.append(min_dis)
                    ours_avg_dis.append(avg_dis)
                    # metrics for p2p
                    composed_region = (region_mask[:, 0, :, :, None]*img_p2p + (
                        1-region_mask[:, 0, :, :, None])*plain_background).round().astype('uint8')
                    rgb_np = rgb.permute([0, 2, 3, 1]).cpu().numpy()
                    euc_dis = np.sqrt(
                        ((composed_region/255.-rgb_np)**2).sum(-1))
                    min_dis = np.min(euc_dis)
                    euc_dis = np.sqrt(((img_p2p/255.-rgb_np)**2).sum(-1))
                    avg_dis = (
                        euc_dis * region_mask[:, 0, :, :]).sum() / region_mask[:, 0, :, :].sum()
                    p2p_min_dis.append(min_dis)
                    p2p_avg_dis.append(avg_dis)
            print('Min dis. N: %d, plain: %.3f±%.3f, ours: %.3f±%.3f, p2p: %.3f±%.3f' % (
                len(ours_min_dis), np.mean(plain_min_dis),
                np.std(plain_min_dis), np.mean(ours_min_dis),
                np.std(ours_min_dis), np.mean(p2p_min_dis),
                np.std(p2p_min_dis)))
            print('Avg dis. N: %d, plain: %.3f±%.3f, ours: %.3f±%.3f, p2p: %.3f±%.3f' % (
                len(ours_avg_dis), np.mean(plain_avg_dis),
                np.std(plain_avg_dis), np.mean(ours_avg_dis),
                np.std(ours_avg_dis), np.mean(p2p_avg_dis),
                np.std(p2p_avg_dis)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace',  type=str, default="results",
                        help="workspace to store result")
    # rich text configs
    parser.add_argument('--foldername', type=str, default="eval",
                        help="folder name under workspace")
    parser.add_argument('--category', type=str, default="common",
                        choices=['common', 'html', 'rgb'],  help="color category")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--save_img', action='store_true',
                        help="save the generated image")
    parser.add_argument('--load_previous', action='store_true',
                        help="Load from previous result, and compute the metrics")
    args = parser.parse_args()

    save_path = os.path.join(args.workspace, args.foldername)
    os.makedirs(save_path, exist_ok=True)

    main(args)
