import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision

from pathlib import Path


def split_attention_maps_over_steps(attention_maps):
    r"""Function for splitting attention maps over steps.
    Args:
        attention_maps (dict): Dictionary of attention maps.
        sampler_order (int): Order of the sampler.
    """
    # This function splits attention maps into unconditional and conditional score and over steps

    attention_maps_cond = dict()    # Maps corresponding to conditional score
    attention_maps_uncond = dict()  # Maps corresponding to unconditional score

    for layer in attention_maps.keys():

        for step_num in range(len(attention_maps[layer])):
            if step_num not in attention_maps_cond:
                attention_maps_cond[step_num] = dict()
                attention_maps_uncond[step_num] = dict()

            attention_maps_uncond[step_num].update(
                {layer: attention_maps[layer][step_num][:1]})
            attention_maps_cond[step_num].update(
                {layer: attention_maps[layer][step_num][1:2]})

    return attention_maps_cond, attention_maps_uncond


def plot_attention_maps(atten_map_list, obj_tokens, save_dir, seed, tokens_vis=None):
    atten_names = ['presoftmax', 'postsoftmax', 'postsoftmax_erosion']
    for i, (attn_map, obj_token) in enumerate(zip(atten_map_list, obj_tokens)):
        n_obj = len(attn_map)
        plt.figure()
        plt.clf()

        fig, axs = plt.subplots(
            ncols=n_obj+1, gridspec_kw=dict(width_ratios=[1 for _ in range(n_obj)]+[0.1]))

        fig.set_figheight(3)
        fig.set_figwidth(3*n_obj+0.1)

        cmap = plt.get_cmap('OrRd')

        vmax = 0
        vmin = 1
        for tid in range(n_obj):
            attention_map_cur = attn_map[tid]
            vmax = max(vmax, float(attention_map_cur.max()))
            vmin = min(vmin, float(attention_map_cur.min()))

        for tid in range(n_obj):
            sns.heatmap(
                attn_map[tid][0], annot=False, cbar=False, ax=axs[tid],
                cmap=cmap, vmin=vmin, vmax=vmax
            )
            axs[tid].set_axis_off()
            if tokens_vis is not None:
                if tid == n_obj-1:
                    axs_xlabel = 'other tokens'
                else:
                    axs_xlabel = ''
                    for token_id in obj_tokens[tid]:
                        axs_xlabel += ' ' + tokens_vis[token_id.item() -
                                                       1][:-len('</w>')]
                axs[tid].set_title(axs_xlabel)

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, cax=axs[-1])

        fig.tight_layout()

        canvas = fig.canvas
        canvas.draw()
        width, height = canvas.get_width_height()
        img = np.frombuffer(canvas.tostring_rgb(),
                            dtype='uint8').reshape((height, width, 3))

        plt.savefig(os.path.join(
            save_dir, 'token_maps_seed%d_%s.png' % (seed, atten_names[i])), dpi=100)
        plt.close('all')
        return img


def get_token_maps(attention_maps, save_dir, width, height, obj_tokens, seed=0, tokens_vis=None,
                   preprocess=False, retur_vis=False):
    r"""Function to visualize attention maps.
    Args:
        save_dir (str): Path to save attention maps
        batch_size (int): Batch size
        sampler_order (int): Sampler order
    """

    # Split attention maps over steps
    attention_maps_cond, _ = split_attention_maps_over_steps(
        attention_maps
    )

    selected_layers = [
        # 'down_blocks.0.attentions.0.transformer_blocks.0.attn2',
        # 'down_blocks.0.attentions.1.transformer_blocks.0.attn2',
        'down_blocks.1.attentions.0.transformer_blocks.0.attn2',
        # 'down_blocks.1.attentions.1.transformer_blocks.0.attn2',
        'down_blocks.2.attentions.0.transformer_blocks.0.attn2',
        'down_blocks.2.attentions.1.transformer_blocks.0.attn2',
        'mid_block.attentions.0.transformer_blocks.0.attn2',
        'up_blocks.1.attentions.0.transformer_blocks.0.attn2',
        'up_blocks.1.attentions.1.transformer_blocks.0.attn2',
        'up_blocks.1.attentions.2.transformer_blocks.0.attn2',
        # 'up_blocks.2.attentions.0.transformer_blocks.0.attn2',
        'up_blocks.2.attentions.1.transformer_blocks.0.attn2',
        # 'up_blocks.2.attentions.2.transformer_blocks.0.attn2',
        # 'up_blocks.3.attentions.0.transformer_blocks.0.attn2',
        # 'up_blocks.3.attentions.1.transformer_blocks.0.attn2',
        # 'up_blocks.3.attentions.2.transformer_blocks.0.attn2'
    ]

    nsteps = len(attention_maps_cond)
    hw_ori = width * height

    attention_maps = []
    for obj_token in obj_tokens:
        attention_maps.append([])

    for step_num in range(nsteps):
        attention_maps_cur = attention_maps_cond[step_num]

        for layer in attention_maps_cur.keys():
            if step_num < 10 or layer not in selected_layers:
                continue

            attention_ind = attention_maps_cur[layer].cpu()

            # Attention maps are of shape [batch_size, nkeys, 77]
            # since they are averaged out while collecting from hooks to save memory.
            # Now split the heads from batch dimension
            bs, hw, nclip = attention_ind.shape
            down_ratio = np.sqrt(hw_ori // hw)
            width_cur = int(width // down_ratio)
            height_cur = int(height // down_ratio)
            attention_ind = attention_ind.reshape(
                bs, height_cur, width_cur, nclip)
            for obj_id, obj_token in enumerate(obj_tokens):
                if obj_token[0] == -1:
                    attention_map_prev = torch.stack(
                        [attention_maps[i][-1] for i in range(obj_id)]).sum(0)
                    attention_maps[obj_id].append(
                        attention_map_prev.max()-attention_map_prev)
                else:
                    obj_attention_map = attention_ind[:, :, :, obj_token].max(-1, True)[
                        0].permute([3, 0, 1, 2])
                    obj_attention_map = torchvision.transforms.functional.resize(obj_attention_map, (height, width),
                                                                                 interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)
                    attention_maps[obj_id].append(obj_attention_map)

    # average attention maps over steps
    attention_maps_averaged = []
    for obj_id, obj_token in enumerate(obj_tokens):
        if obj_id == len(obj_tokens) - 1:
            attention_maps_averaged.append(
                torch.cat(attention_maps[obj_id]).mean(0))
        else:
            attention_maps_averaged.append(
                torch.cat(attention_maps[obj_id]).mean(0))

    # normalize attention maps into [0, 1]
    attention_maps_averaged_normalized = []
    attention_maps_averaged_sum = torch.cat(attention_maps_averaged).sum(0)
    for obj_id, obj_token in enumerate(obj_tokens):
        attention_maps_averaged_normalized.append(
            attention_maps_averaged[obj_id]/attention_maps_averaged_sum)

    # softmax
    attention_maps_averaged_normalized = (
        torch.cat(attention_maps_averaged)/0.001).softmax(0)
    attention_maps_averaged_normalized = [
        attention_maps_averaged_normalized[i:i+1] for i in range(attention_maps_averaged_normalized.shape[0])]

    if preprocess:
        # it is possible to preprocess the attention maps here
        import skimage
        from skimage.morphology import erosion, square
        selem = square(5)
        attention_maps_averaged_eroded = [erosion(skimage.img_as_float(
            map[0].numpy()*255), selem) for map in attention_maps_averaged_normalized[:2]]
        attention_maps_averaged_eroded = [(torch.from_numpy(map).unsqueeze(
            0)/255. > 0.8).float() for map in attention_maps_averaged_eroded]
        attention_maps_averaged_eroded.append(
            1 - torch.cat(attention_maps_averaged_eroded).sum(0, True))
        token_maps_vis = plot_attention_maps([attention_maps_averaged, attention_maps_averaged_normalized,
                                              attention_maps_averaged_eroded], obj_tokens, save_dir, seed, tokens_vis)
        attention_maps_averaged_eroded = [attn_mask.unsqueeze(1).repeat(
            [1, 4, 1, 1]).cuda() for attn_mask in attention_maps_averaged_eroded]
        token_maps = attention_maps_averaged_eroded
    else:
        token_maps_vis = plot_attention_maps([attention_maps_averaged, attention_maps_averaged_normalized],
                                             obj_tokens, save_dir, seed, tokens_vis)
        attention_maps_averaged_normalized = [attn_mask.unsqueeze(1).repeat(
            [1, 4, 1, 1]).cuda() for attn_mask in attention_maps_averaged_normalized]
        token_maps = attention_maps_averaged_normalized
    if retur_vis:
        return token_maps, token_maps_vis
    else:
        return token_maps
