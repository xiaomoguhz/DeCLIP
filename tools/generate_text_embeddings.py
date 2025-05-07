# Modified from [ViLD](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild)
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import open_clip


def article(name):
  return 'an' if name[0] in 'aeiou' else 'a'

def processed_name(name, rm_dot=False):
  # _ for lvis
  # / for obj365
  res = name.replace('_', ' ').replace('/', ' or ').lower()
  if rm_dot:
    res = res.rstrip('.')
  return res


single_template = [
    'a photo of {article} {}.'
]

multiple_templates = [
    'There is {article} {} in the scene.',
    'There is the {} in the scene.',
    'a photo of {article} {} in the scene.',
    'a photo of the {} in the scene.',
    'a photo of one {} in the scene.',


    'itap of {article} {}.',
    'itap of my {}.',  # itap: I took a picture of
    'itap of the {}.',
    'a photo of {article} {}.',
    'a photo of my {}.',
    'a photo of the {}.',
    'a photo of one {}.',
    'a photo of many {}.',

    'a good photo of {article} {}.',
    'a good photo of the {}.',
    'a bad photo of {article} {}.',
    'a bad photo of the {}.',
    'a photo of a nice {}.',
    'a photo of the nice {}.',
    'a photo of a cool {}.',
    'a photo of the cool {}.',
    'a photo of a weird {}.',
    'a photo of the weird {}.',

    'a photo of a small {}.',
    'a photo of the small {}.',
    'a photo of a large {}.',
    'a photo of the large {}.',

    'a photo of a clean {}.',
    'a photo of the clean {}.',
    'a photo of a dirty {}.',
    'a photo of the dirty {}.',

    'a bright photo of {article} {}.',
    'a bright photo of the {}.',
    'a dark photo of {article} {}.',
    'a dark photo of the {}.',

    'a photo of a hard to see {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of {article} {}.',
    'a low resolution photo of the {}.',
    'a cropped photo of {article} {}.',
    'a cropped photo of the {}.',
    'a close-up photo of {article} {}.',
    'a close-up photo of the {}.',
    'a jpeg corrupted photo of {article} {}.',
    'a jpeg corrupted photo of the {}.',
    'a blurry photo of {article} {}.',
    'a blurry photo of the {}.',
    'a pixelated photo of {article} {}.',
    'a pixelated photo of the {}.',

    'a black and white photo of the {}.',
    'a black and white photo of {article} {}.',

    'a plastic {}.',
    'the plastic {}.',

    'a toy {}.',
    'the toy {}.',
    'a plushie {}.',
    'the plushie {}.',
    'a cartoon {}.',
    'the cartoon {}.',

    'an embroidered {}.',
    'the embroidered {}.',

    'a painting of the {}.',
    'a painting of a {}.',
]


def build_text_embedding_coco(categories, model):
    templates = multiple_templates
    with torch.no_grad():
        zeroshot_weights = []
        attn12_weights = []
        for category in categories:
            texts = [
                template.format(processed_name(category, rm_dot=True), article=article(category))
                for template in templates
            ]
            texts = [
                "This is " + text if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ]
            texts = open_clip.tokenize(texts).cuda()  # tokenize
            text_embeddings = model.encode_text(texts)
            text_attnfeatures, _, _ = model.encode_text_endk(texts, stepk=12, normalize=True)

            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()

            text_attnfeatures = text_attnfeatures.mean(0)
            text_attnfeatures = F.normalize(text_attnfeatures, dim=0)
            attn12_weights.append(text_attnfeatures)
            zeroshot_weights.append(text_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0)
        attn12_weights = torch.stack(attn12_weights, dim=0)

    return zeroshot_weights, attn12_weights


def build_text_embedding_lvis(categories, model, tokenizer):
    templates = multiple_templates

    with torch.no_grad():
        all_text_embeddings = []
        for category in tqdm(categories):
            texts = [
                template.format(
                    processed_name(category, rm_dot=True), article=article(category)
                )
                for template in templates
            ]
            texts = [
                "This is " + text if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ]
            texts = tokenizer(texts).cuda()  # tokenize

            text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()

            all_text_embeddings.append(text_embedding)
        all_text_embeddings = torch.stack(all_text_embeddings, dim=0)

    return all_text_embeddings


# voc_cats = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
#             'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#             'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
#             'tvmonitor')
# text_embeddings, _ = build_text_embedding_coco(voc_cats)
# np.save('datasets/metadata/voc_clip_hand_craft.npy', text_embeddings.cpu().numpy())

import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', default='ViT-L-14-336')
    parser.add_argument('--ann', default='data/coco/annotations/panoptic_val2017.json')
    parser.add_argument('--out_path', default='metadata/coco_panoptic_clip_hand_craft_ViTL14x336.npy')
    parser.add_argument('--pretrained', default='openai')
    parser.add_argument('--cache_dir', default='checkpoints')

    args = parser.parse_args()

    model = open_clip.create_model(
        args.model_version, pretrained=args.pretrained, cache_dir=args.cache_dir
    )
    tokenizer = open_clip.get_tokenizer(args.model_version)
    model.cuda()

    print('Loading', args.ann)
    data = json.load(open(args.ann, 'r'))
    cat_names = [x['name'] for x in \
                 sorted(data['categories'], key=lambda x: x['id'])]
    out_path = args.out_path
    text_embeddings = build_text_embedding_lvis(cat_names, model, tokenizer)
    np.save(out_path, text_embeddings.cpu().numpy())
