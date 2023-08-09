import sys
sys.path.append('.')

from utils.attention_utils import get_token_maps
from utils.richtext_utils import seed_everything
from utils import ptp_utils, clip_utils
from models.region_diffusion import RegionDiffusion
from diffusers import StableDiffusionPipeline
import torchvision
import imageio
import numpy as np
import argparse
import torch
import os


NUM_DIFFUSION_STEPS = 41
GUIDANCE_SCALE = 8.5

text_prompt_all = [
    'A garden with a mountain in the distance.',
    'A fountain in front of an castle.',
    'A cat sitting on a meadow.',
    'A lighthouse among the turbulent waves in the night.',
    'A stream train on the mountain side.',
    'A cactus standing in the desert.',
    'A dog sitting on a beach.',
    'A solitary rowboat tethered on a serene pond.',
    'A house on a rocky mountain.',
    'A rustic windmill on a grassy hill.',
]
text_prompts_all = [
    ['garden', 'mountain'],
    ['fountain', 'castle'],
    ['cat', 'meadow'],
    ['lighthouse', 'turbulent waves'],
    ['stream train', 'mountain side'],
    ['cactus', 'desert'],
    ['dog', 'beach'],
    ['rowboat', 'pond'],
    ['house', 'mountain'],
    ['rustic', 'hill'],
]
styles = [
    'Claud Monet, impressionism, oil on canvas',
    'Ukiyoe',
    'Cyber Punk, futuristic, blade runner, william gibson, trending on artstation hq',
    'Pop Art, masterpiece, andy warhol',
    'Vincent Van Gogh',
    'Pixel Art, 8 bits, 16 bits',
    'Abstract Cubism, Pablo Picasso'
]


def main(args):
    negative_text = ''
    text_format_dict = {}
    height = 512
    width = 512
    init_seed = args.seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    region_model = RegionDiffusion(device)
    clip_model = clip_utils.CLIPEncoder()
    ldm_stable = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5").to(device)
    tokenizer = ldm_stable.tokenizer

    ours_clip_scores = []
    p2p_clip_scores = []
    ours_region_clip_scores = [[], []]
    p2p_region_clip_scores = [[], []]

    for seed in range(init_seed, init_seed+3):
        seed_everything(seed)
        latent = torch.randn((1, 4, height // 8, width // 8), device='cuda')

        for text_prompt, text_prompts in zip(text_prompt_all, text_prompts_all):
            base_name = '_'.join(text_prompts)
            region_model.register_tokenmap_hooks()
            seed_everything(seed)
            img = region_model.produce_attn_maps([text_prompt], [negative_text],
                                                 height=height, width=width, num_inference_steps=NUM_DIFFUSION_STEPS,
                                                 guidance_scale=GUIDANCE_SCALE, latents=latent)
            obj_token_ids = []
            base_tokens = region_model.tokenizer._tokenize(text_prompt)
            for prompt in text_prompts:
                obj_token_ids.append([])
                style_tokens = region_model.tokenizer._tokenize(prompt)
                for style_token in style_tokens:
                    obj_token_ids[-1].append(base_tokens.index(style_token)+1)
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
            if args.save_img:
                imageio.imwrite(os.path.join(
                    save_path, 'ours_%s.png' % (base_name)), img[0])
            region_model.remove_tokenmap_hooks()
            for style1 in styles:
                for style2 in styles:
                    if style1 == style2:
                        continue
                    style_name = '_'.join([style1, style2])
                    ours_name = os.path.join(
                        save_path, 'ours_%s_%s.png' % (base_name, style_name))
                    p2p_name = os.path.join(
                        save_path, 'p2p_%s_%s.png' % (base_name, style_name))
                    text_prompts_rich = [
                        prompt + f' in the style of {style}' for prompt, style in zip(text_prompts, [style1, style2])]
                    text_prompts_rich.append(text_prompt)
                    if not args.load_previous:
                        seed_everything(seed)
                        img_ours = region_model.prompt_to_img(text_prompts_rich, [negative_text],
                                                            height=height, width=width, num_inference_steps=NUM_DIFFUSION_STEPS,
                                                            guidance_scale=GUIDANCE_SCALE, text_format_dict=text_format_dict, latents=latent,
                                                            use_guidance=False)
                        if args.save_img:
                            imageio.imwrite(ours_name, img_ours[0])
                        img_ours = img_ours.astype(float)
                        text_prompt_p2p = text_prompt.replace(text_prompts[0], text_prompts_rich[0]).replace(
                            text_prompts[1], text_prompts_rich[1])
                        text_prompts_p2p = [text_prompt, text_prompt_p2p]
                        controller = ptp_utils.AttentionRefine(text_prompts_p2p, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
                                                               self_replace_steps=.4, tokenizer=tokenizer)
                        seed_everything(seed)
                        img_p2p, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, text_prompts_p2p, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS,
                                                                       guidance_scale=GUIDANCE_SCALE, generator=None, low_resource=False)
                        if args.save_img:
                            imageio.imwrite(os.path.join(
                                save_path, 'p2p_%s.png' % (base_name)), img_p2p[0])
                            imageio.imwrite(p2p_name, img_p2p[1])
                        img_p2p = img_p2p[1]
                    else:
                        img_ours = np.array(imageio.imread(ours_name))
                        img_p2p = np.array(imageio.imread(p2p_name))

                    ours_imgs = []
                    p2p_imgs = []
                    black_background = np.zeros_like(img_ours)
                    for region_id, region_mask in enumerate(region_masks[:-1]):
                        composed_region = (region_mask[:, 0, :, :, None]*img_ours + (
                            1-region_mask[:, 0, :, :, None])*black_background).round().astype('uint8')
                        ours_imgs.append(composed_region)
                        composed_region = (region_mask[:, 0, :, :, None]*img_p2p + (
                            1-region_mask[:, 0, :, :, None])*black_background).round().astype('uint8')
                        p2p_imgs.append(composed_region)

                    for prompt_id in range(2):
                        ours_clip_score = (clip_model.get_clip_score(
                            text_prompts_rich[prompt_id], ours_imgs[prompt_id][0])).item()
                        p2p_clip_score = (clip_model.get_clip_score(
                            text_prompts_rich[prompt_id], p2p_imgs[prompt_id][0])).item()
                        ours_clip_scores.append(ours_clip_score)
                        ours_region_clip_scores[prompt_id].append(
                            ours_clip_score)
                        p2p_clip_scores.append(p2p_clip_score)
                        p2p_region_clip_scores[prompt_id].append(
                            p2p_clip_score)

                    print('N: %d, ours: %.4f±%.4f, p2p: %.4f±%.4f' % (len(ours_clip_scores), np.mean(
                        ours_clip_scores), np.std(ours_clip_scores), np.mean(p2p_clip_scores), np.std(p2p_clip_scores)))
                    print('Region 1, ours: %.4f±%.4f, p2p: %.4f±%.4f' % (np.mean(ours_region_clip_scores[0]), np.std(
                        ours_region_clip_scores[0]), np.mean(p2p_region_clip_scores[0]), np.std(p2p_region_clip_scores[0])))
                    print('Region 2, ours: %.4f±%.4f, p2p: %.4f±%.4f' % (np.mean(ours_region_clip_scores[1]), np.std(
                        ours_region_clip_scores[1]), np.mean(p2p_region_clip_scores[1]), np.std(p2p_region_clip_scores[1])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace',  type=str, default="results",
                        help="workspace to store result")
    # rich text configs
    parser.add_argument('--foldername', type=str, default="eval",
                        help="folder name under workspace")
    parser.add_argument('--save_img', action='store_true',
                        help="save the generated image")
    parser.add_argument('--seed',       type=int,
                        default=0,                help="random seed")
    parser.add_argument('--load_previous',
                        action='store_true',     help="Load from previous result")
    args = parser.parse_args()

    save_path = os.path.join(args.workspace, args.foldername)
    os.makedirs(save_path, exist_ok=True)

    main(args)
