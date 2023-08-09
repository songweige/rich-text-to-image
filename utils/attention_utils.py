import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision

from utils.richtext_utils import seed_everything
from sklearn.cluster import SpectralClustering

SelfAttentionLayers = [
    'down_blocks.0.attentions.0.transformer_blocks.0.attn1',
    'down_blocks.0.attentions.1.transformer_blocks.0.attn1',
    'down_blocks.1.attentions.0.transformer_blocks.0.attn1',
    'down_blocks.1.attentions.1.transformer_blocks.0.attn1',
    'down_blocks.2.attentions.0.transformer_blocks.0.attn1',
    'down_blocks.2.attentions.1.transformer_blocks.0.attn1',
    'mid_block.attentions.0.transformer_blocks.0.attn1',
    'up_blocks.1.attentions.0.transformer_blocks.0.attn1',
    'up_blocks.1.attentions.1.transformer_blocks.0.attn1',
    'up_blocks.1.attentions.2.transformer_blocks.0.attn1',
    'up_blocks.2.attentions.0.transformer_blocks.0.attn1',
    'up_blocks.2.attentions.1.transformer_blocks.0.attn1',
    'up_blocks.2.attentions.2.transformer_blocks.0.attn1',
    'up_blocks.3.attentions.0.transformer_blocks.0.attn1',
    'up_blocks.3.attentions.1.transformer_blocks.0.attn1',
    'up_blocks.3.attentions.2.transformer_blocks.0.attn1',
]


CrossAttentionLayers = [
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


CrossAttentionLayers_XL = [
    'down_blocks.2.attentions.1.transformer_blocks.3.attn2',
    'down_blocks.2.attentions.1.transformer_blocks.4.attn2',
    'mid_block.attentions.0.transformer_blocks.0.attn2',
    'mid_block.attentions.0.transformer_blocks.1.attn2',
    'mid_block.attentions.0.transformer_blocks.2.attn2',
    'mid_block.attentions.0.transformer_blocks.3.attn2',
    'up_blocks.0.attentions.0.transformer_blocks.1.attn2',
    'up_blocks.0.attentions.0.transformer_blocks.2.attn2',
    'up_blocks.0.attentions.0.transformer_blocks.3.attn2',
    'up_blocks.0.attentions.0.transformer_blocks.4.attn2',
    'up_blocks.0.attentions.0.transformer_blocks.5.attn2',
    'up_blocks.0.attentions.0.transformer_blocks.6.attn2',
    'up_blocks.0.attentions.0.transformer_blocks.7.attn2',
    'up_blocks.1.attentions.0.transformer_blocks.0.attn2'
]


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
    for i, attn_map in enumerate(atten_map_list):
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
        canvas = fig.canvas
        canvas.draw()
        width, height = canvas.get_width_height()
        img = np.frombuffer(canvas.tostring_rgb(),
                            dtype='uint8').reshape((height, width, 3))

        fig.tight_layout()

        plt.savefig(os.path.join(
            save_dir, 'average_seed%d_attn%d.png' % (seed, i)), dpi=100)
        plt.close('all')
    return img


def get_token_maps_deprecated(attention_maps, save_dir, width, height, obj_tokens, seed=0, tokens_vis=None):
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

    nsteps = len(attention_maps_cond)
    hw_ori = width * height

    attention_maps = []
    for obj_token in obj_tokens:
        attention_maps.append([])

    for step_num in range(nsteps):
        attention_maps_cur = attention_maps_cond[step_num]

        for layer in attention_maps_cur.keys():
            if step_num < 10 or layer not in CrossAttentionLayers:
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

    token_maps_vis = plot_attention_maps([attention_maps_averaged, attention_maps_averaged_normalized],
                                         obj_tokens, save_dir, seed, tokens_vis)
    attention_maps_averaged_normalized = [attn_mask.unsqueeze(1).repeat(
        [1, 4, 1, 1]).cuda() for attn_mask in attention_maps_averaged_normalized]
    return attention_maps_averaged_normalized, token_maps_vis


def get_token_maps(selfattn_maps, crossattn_maps, n_maps, save_dir, width, height, obj_tokens, seed=0, tokens_vis=None,
                   preprocess=False, segment_threshold=0.3, num_segments=5, return_vis=False, save_attn=False):
    r"""Function to visualize attention maps.
    Args:
        save_dir (str): Path to save attention maps
        batch_size (int): Batch size
        sampler_order (int): Sampler order
    """

    # create the segmentation mask using self-attention maps
    resolution = 32
    attn_maps_1024 = {8: [], 16: [], 32: [], 64: []}
    for attn_map in selfattn_maps.values():
        resolution_map = np.sqrt(attn_map.shape[1]).astype(int)
        if resolution_map != resolution:
            continue
        attn_map = attn_map.reshape(
            1, resolution_map, resolution_map, resolution_map**2).permute([3, 0, 1, 2]).float()
        attn_map = torch.nn.functional.interpolate(attn_map, (resolution, resolution),
                                                mode='bicubic', antialias=True)
        attn_maps_1024[resolution_map].append(attn_map.permute([1, 2, 3, 0]).reshape(
            1, resolution**2, resolution_map**2))
    attn_maps_1024 = torch.cat([torch.cat(v).mean(0).cpu()
                                for v in attn_maps_1024.values() if len(v) > 0], -1).numpy()
    if save_attn:
        print('saving self-attention maps...', attn_maps_1024.shape)
        torch.save(torch.from_numpy(attn_maps_1024),
                   'results/maps/selfattn_maps.pth')
    seed_everything(seed)
    sc = SpectralClustering(num_segments, affinity='precomputed', n_init=100,
                            assign_labels='kmeans')
    clusters = sc.fit_predict(attn_maps_1024)
    clusters = clusters.reshape(resolution, resolution)
    fig = plt.figure()
    plt.imshow(clusters)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'segmentation_k%d_seed%d.jpg' % (num_segments, seed)),
                bbox_inches='tight', pad_inches=0)
    if return_vis:
        canvas = fig.canvas
        canvas.draw()
        cav_width, cav_height = canvas.get_width_height()
        segments_vis = np.frombuffer(canvas.tostring_rgb(),
                                     dtype='uint8').reshape((cav_height, cav_width, 3))

    plt.close()

    # label the segmentation mask using cross-attention maps
    cross_attn_maps_1024 = []
    for attn_map in crossattn_maps.values():
        resolution_map = np.sqrt(attn_map.shape[1]).astype(int)
        attn_map = attn_map.reshape(
            1, resolution_map, resolution_map, -1).permute([0, 3, 1, 2]).float()
        attn_map = torch.nn.functional.interpolate(attn_map, (resolution, resolution),
                                                   mode='bicubic', antialias=True)
        cross_attn_maps_1024.append(attn_map.permute([0, 2, 3, 1]))

    cross_attn_maps_1024 = torch.cat(
        cross_attn_maps_1024).mean(0).cpu().numpy()
    if save_attn:
        print('saving cross-attention maps...', cross_attn_maps_1024.shape)
        torch.save(torch.from_numpy(cross_attn_maps_1024),
                   'results/maps/crossattn_maps.pth')
    normalized_span_maps = []
    for token_ids in obj_tokens:
        span_token_maps = cross_attn_maps_1024[:, :, token_ids.numpy()]
        normalized_span_map = np.zeros_like(span_token_maps)
        for i in range(span_token_maps.shape[-1]):
            curr_noun_map = span_token_maps[:, :, i]
            normalized_span_map[:, :, i] = (
                curr_noun_map - np.abs(curr_noun_map.min())) / (curr_noun_map.max()-curr_noun_map.min())
        normalized_span_maps.append(normalized_span_map)
    foreground_token_maps = [np.zeros([clusters.shape[0], clusters.shape[1]]).squeeze(
    ) for normalized_span_map in normalized_span_maps]
    background_map = np.zeros([clusters.shape[0], clusters.shape[1]]).squeeze()
    for c in range(num_segments):
        cluster_mask = np.zeros_like(clusters)
        cluster_mask[clusters == c] = 1.
        is_foreground = False
        for normalized_span_map, foreground_nouns_map, token_ids in zip(normalized_span_maps, foreground_token_maps, obj_tokens):
            score_maps = [cluster_mask * normalized_span_map[:, :, i]
                          for i in range(len(token_ids))]
            scores = [score_map.sum() / cluster_mask.sum()
                      for score_map in score_maps]
            if max(scores) > segment_threshold:
                foreground_nouns_map += cluster_mask
                is_foreground = True
        if not is_foreground:
            background_map += cluster_mask
    foreground_token_maps.append(background_map)

    # resize the token maps and visualization
    resized_token_maps = torch.cat([torch.nn.functional.interpolate(torch.from_numpy(token_map).unsqueeze(0).unsqueeze(
        0), (height, width), mode='bicubic', antialias=True)[0] for token_map in foreground_token_maps]).clamp(0, 1)

    resized_token_maps = resized_token_maps / \
        (resized_token_maps.sum(0, True)+1e-8)
    resized_token_maps = [token_map.unsqueeze(
        0) for token_map in resized_token_maps]
    foreground_token_maps = [token_map[None, :, :]
                             for token_map in foreground_token_maps]
    token_maps_vis = plot_attention_maps([foreground_token_maps, resized_token_maps], obj_tokens,
                                         save_dir, seed, tokens_vis)
    resized_token_maps = [token_map.unsqueeze(1).repeat(
        [1, 4, 1, 1]).to(attn_map.dtype).cuda() for token_map in resized_token_maps]
    if return_vis:
        return resized_token_maps, segments_vis, token_maps_vis
    else:
        return resized_token_maps
