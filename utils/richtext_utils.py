import os
import json
import torch
import random
import numpy as np

COLORS = {
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
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def hex_to_rgb(hex_string, return_nearest_color=False):
    r"""
    Covert Hex triplet to RGB triplet.
    """
    # Remove '#' symbol if present
    hex_string = hex_string.lstrip('#')
    # Convert hex values to integers
    red = int(hex_string[0:2], 16)
    green = int(hex_string[2:4], 16)
    blue = int(hex_string[4:6], 16)
    rgb = torch.FloatTensor((red, green, blue))[None, :, None, None]/255.
    if return_nearest_color:
        nearest_color = find_nearest_color(rgb)
        return rgb.cuda(), nearest_color
    return rgb.cuda()


def find_nearest_color(rgb):
    r"""
    Find the nearest neighbor color given the RGB value.
    """
    if isinstance(rgb, list) or isinstance(rgb, tuple):
        rgb = torch.FloatTensor(rgb)[None, :, None, None]/255.
    color_distance = torch.FloatTensor([np.linalg.norm(
        rgb - torch.FloatTensor(COLORS[color])[None, :, None, None]/255.) for color in COLORS.keys()])
    nearest_color = list(COLORS.keys())[torch.argmin(color_distance).item()]
    return nearest_color


def font2style(font):
    r"""
    Convert the font name to the style name.
    """
    return {'mirza': 'Claud Monet, impressionism, oil on canvas',
            'roboto': 'Ukiyoe',
            'cursive': 'Cyber Punk, futuristic, blade runner, william gibson, trending on artstation hq',
            'sofia': 'Pop Art, masterpiece, andy warhol',
            'slabo': 'Vincent Van Gogh',
            'inconsolata': 'Pixel Art, 8 bits, 16 bits',
            'ubuntu': 'Rembrandt',
            'Monoton': 'neon art, colorful light, highly details, octane render',
            'Akronim': 'Abstract Cubism, Pablo Picasso', }[font]


def parse_json(json_str):
    r"""
    Convert the JSON string to attributes.
    """
    # initialze region-base attributes.
    base_text_prompt = ''
    style_text_prompts = []
    footnote_text_prompts = []
    footnote_target_tokens = []
    color_text_prompts = []
    color_rgbs = []
    color_names = []
    size_text_prompts_and_sizes = []

    # parse the attributes from JSON.
    prev_style = None
    prev_color_rgb = None
    use_grad_guidance = False
    for span in json_str['ops']:
        text_prompt = span['insert'].rstrip('\n')
        base_text_prompt += span['insert'].rstrip('\n')
        if text_prompt == ' ':
            continue
        if 'attributes' in span:
            if 'font' in span['attributes']:
                style = font2style(span['attributes']['font'])
                if prev_style == style:
                    prev_text_prompt = style_text_prompts[-1].split('in the style of')[
                        0]
                    style_text_prompts[-1] = prev_text_prompt + \
                        ' ' + text_prompt + f' in the style of {style}'
                else:
                    style_text_prompts.append(
                        text_prompt + f' in the style of {style}')
                prev_style = style
            else:
                prev_style = None
            if 'link' in span['attributes']:
                footnote_text_prompts.append(span['attributes']['link'])
                footnote_target_tokens.append(text_prompt)
            font_size = 1
            if 'size' in span['attributes'] and 'strike' not in span['attributes']:
                font_size = float(span['attributes']['size'][:-2])/3.
            elif 'size' in span['attributes'] and 'strike' in span['attributes']:
                font_size = -float(span['attributes']['size'][:-2])/3.
            elif 'size' not in span['attributes'] and 'strike' not in span['attributes']:
                font_size = 1
            if 'color' in span['attributes']:
                use_grad_guidance = True
                color_rgb, nearest_color = hex_to_rgb(
                    span['attributes']['color'], True)
                if prev_color_rgb == color_rgb:
                    prev_text_prompt = color_text_prompts[-1]
                    color_text_prompts[-1] = prev_text_prompt + \
                        ' ' + text_prompt
                else:
                    color_rgbs.append(color_rgb)
                    color_names.append(nearest_color)
                    color_text_prompts.append(text_prompt)
            if font_size != 1:
                size_text_prompts_and_sizes.append([text_prompt, font_size])
    return base_text_prompt, style_text_prompts, footnote_text_prompts, footnote_target_tokens,\
        color_text_prompts, color_names, color_rgbs, size_text_prompts_and_sizes, use_grad_guidance


def get_region_diffusion_input(model, base_text_prompt, style_text_prompts, footnote_text_prompts,
                               footnote_target_tokens, color_text_prompts, color_names):
    r"""
    Algorithm 1 in the paper.
    """
    region_text_prompts = []
    region_target_token_ids = []
    base_tokens = model.tokenizer._tokenize(base_text_prompt)
    # process the style text prompt
    for text_prompt in style_text_prompts:
        region_text_prompts.append(text_prompt)
        region_target_token_ids.append([])
        style_tokens = model.tokenizer._tokenize(
            text_prompt.split('in the style of')[0])
        for style_token in style_tokens:
            region_target_token_ids[-1].append(
                base_tokens.index(style_token)+1)

    # process the complementary text prompt
    for footnote_text_prompt, text_prompt in zip(footnote_text_prompts, footnote_target_tokens):
        region_target_token_ids.append([])
        region_text_prompts.append(footnote_text_prompt)
        style_tokens = model.tokenizer._tokenize(text_prompt)
        for style_token in style_tokens:
            region_target_token_ids[-1].append(
                base_tokens.index(style_token)+1)

    # process the color text prompt
    for color_text_prompt, color_name in zip(color_text_prompts, color_names):
        region_target_token_ids.append([])
        region_text_prompts.append(color_name+' '+color_text_prompt)
        style_tokens = model.tokenizer._tokenize(color_text_prompt)
        for style_token in style_tokens:
            region_target_token_ids[-1].append(
                base_tokens.index(style_token)+1)

    # process the remaining tokens without any attributes
    region_text_prompts.append(base_text_prompt)
    region_target_token_ids_all = [
        id for ids in region_target_token_ids for id in ids]
    target_token_ids_rest = [id for id in range(
        1, len(base_tokens)+1) if id not in region_target_token_ids_all]
    region_target_token_ids.append(target_token_ids_rest)

    region_target_token_ids = [torch.LongTensor(
        obj_token_id) for obj_token_id in region_target_token_ids]
    return region_text_prompts, region_target_token_ids, base_tokens


def get_attention_control_input(model, base_tokens, size_text_prompts_and_sizes):
    r"""
    Control the token impact using font sizes.
    """
    word_pos = []
    font_sizes = []
    for text_prompt, font_size in size_text_prompts_and_sizes:
        size_tokens = model.tokenizer._tokenize(text_prompt)
        for size_token in size_tokens:
            word_pos.append(base_tokens.index(size_token)+1)
            font_sizes.append(font_size)
    if len(word_pos) > 0:
        word_pos = torch.LongTensor(word_pos).cuda()
        font_sizes = torch.FloatTensor(font_sizes).cuda()
    else:
        word_pos = None
        font_sizes = None
    text_format_dict = {
        'word_pos': word_pos,
        'font_size': font_sizes,
    }
    return text_format_dict


def get_gradient_guidance_input(model, base_tokens, color_text_prompts, color_rgbs, text_format_dict,
                                guidance_start_step=999, color_guidance_weight=1):
    r"""
    Control the token impact using font sizes.
    """
    color_target_token_ids = []
    for text_prompt in color_text_prompts:
        color_target_token_ids.append([])
        color_tokens = model.tokenizer._tokenize(text_prompt)
        for color_token in color_tokens:
            color_target_token_ids[-1].append(base_tokens.index(color_token)+1)
    color_target_token_ids_all = [
        id for ids in color_target_token_ids for id in ids]
    color_target_token_ids_rest = [id for id in range(
        1, len(base_tokens)+1) if id not in color_target_token_ids_all]
    color_target_token_ids.append(color_target_token_ids_rest)
    color_target_token_ids = [torch.LongTensor(
        obj_token_id) for obj_token_id in color_target_token_ids]

    text_format_dict['target_RGB'] = color_rgbs
    text_format_dict['guidance_start_step'] = guidance_start_step
    text_format_dict['color_guidance_weight'] = color_guidance_weight
    return text_format_dict, color_target_token_ids
