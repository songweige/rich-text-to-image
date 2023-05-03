import os
import torch
import imageio
import argparse
from models.region_diffusion import RegionDiffusion
from utils.attention_utils import get_token_maps
from utils.richtext_utils import seed_everything


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str,
                        default='results/visualize_token_maps')
    parser.add_argument('--text_prompt', type=str,
                        default='a camera on a tripod taking a picture of a cat.')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--token_ids', type=int, nargs='*',
                        default=None, help="token ids to visualize")
    parser.add_argument('--segment_threshold', type=float, default=0.4)
    parser.add_argument('--num_segments', type=int, default=5)
    args = parser.parse_args()
    seed = args.seed
    seed_everything(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RegionDiffusion(device)

    save_path = args.run_dir
    os.makedirs(save_path, exist_ok=True)
    negative_text = ''
    model.register_tokenmap_hooks()
    base_tokens = model.tokenizer._tokenize(args.text_prompt)
    obj_token_ids = [torch.LongTensor([obj_token_id+1])
                     for obj_token_id in args.token_ids]
    img = model.produce_attn_maps([args.text_prompt], [negative_text],
                                     height=512, width=512, num_inference_steps=41,
                                     guidance_scale=8.5)
    _ = get_token_maps(
        model.selfattn_maps, model.crossattn_maps, model.n_maps, save_path,
                                    512//8, 512//8, obj_token_ids, seed,
                                    base_tokens, segment_threshold=0.45, num_segments=8)
    imageio.imwrite(os.path.join(save_path, 'seed%d.png' % (seed)), img[0])
