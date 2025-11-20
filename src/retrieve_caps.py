import json
from tqdm import tqdm
from transformers import AutoTokenizer
import clip
import torch
import faiss
import os
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_coco_data(coco_data_path):
    """We load in all images and only the train captions."""

    annotations = json.load(open(coco_data_path))['images']
    images = []
    captions = []
    for item in annotations:
        if item['split'] == 'restval':
             item['split'] = 'train'
        if item['split'] == 'train':
            for sentence in item['sentences']:
                captions.append({'image_id': item['cocoid'],  'caption': ' '.join(sentence['tokens'])})
        images.append({'image_id': item['cocoid'], 'file_name': item['filename'].split('_')[-1]})
 
    return images, captions

def filter_captions(data):

    decoder_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    bs = 512

    image_ids = [d['image_id'] for d in data]
    caps = [d['caption'] for d in data]
    encodings = []
    for idx in range(0, len(data), bs):
        encodings += tokenizer.batch_encode_plus(caps[idx:idx+bs], return_tensors='np', padding=True)['input_ids'].tolist()
    
    filtered_image_ids, filtered_captions = [], []

    assert len(image_ids) == len(caps) and len(caps) == len(encodings)
    for image_id, cap, encoding in zip(image_ids, caps, encodings):
        if len(encoding) <= 25:
            filtered_image_ids.append(image_id)
            filtered_captions.append(cap)

    return filtered_image_ids, filtered_captions

def encode_captions(captions, model, device):

    bs = 256
    encoded_captions = []

    for idx in tqdm(range(0, len(captions), bs)):
        with torch.no_grad():
            input_ids = clip.tokenize(captions[idx:idx+bs]).to(device)
            encoded_captions.append(model.encode_text(input_ids).cpu().numpy())

    encoded_captions = np.concatenate(encoded_captions)

    return encoded_captions

def encode_images(images, image_path, model, feature_extractor, device):

    image_ids = [i['image_id'] for i in images]
    
    bs = 64	
    image_features = []
    
    for idx in tqdm(range(0, len(images), bs)):
        image_input = [feature_extractor(Image.open(os.path.join(image_path, i['file_name'])))
                                                                    for i in images[idx:idx+bs]]
        with torch.no_grad():
            image_features.append(model.encode_image(torch.tensor(np.stack(image_input)).to(device)).cpu().numpy())

    image_features = np.concatenate(image_features)

    return image_ids, image_features

def get_nns(captions, images, k=15):
    xq = images.astype(np.float32)
    xb = captions.astype(np.float32)
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    faiss.normalize_L2(xq)
    D, I = index.search(xq, k) 

    return index, I

def filter_nns(nns, xb_image_ids, captions, xq_image_ids):
    """ We filter out nearest neighbors which are actual captions for the query image, keeping 7 neighbors per image."""
    retrieved_captions = {}
    for nns_list, image_id in zip(nns, xq_image_ids):
        good_nns = []
        for nn in nns_list:
            if xb_image_ids[nn] == image_id:
                continue
            good_nns.append(captions[nn])
            if len(good_nns) == 7:
                break
        assert len(good_nns) == 7
        retrieved_captions[image_id] = good_nns
    return retrieved_captions

def load_datastore(datastore_dir):
    """
    Expect files in datastore_dir:
      - index (faiss) e.g. 'coco_index' (binary)
      - captions json e.g. 'coco_index_captions.json' (list of captions in same order as embeddings)
      - optionally text pooled embeddings npy e.g. 'clip_text_pooled.npy' (N, D)
    Returns: index, captions_list, caption_embs (numpy array or None)
    """
    idx_path = os.path.join(datastore_dir, "coco_index")
    caps_json = os.path.join(datastore_dir, "coco_index_captions.json")
    emb_path = os.path.join(datastore_dir, "clip_text_pooled.npy")

    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"faiss index not found at {idx_path}")
    index = faiss.read_index(idx_path)
    captions = json.load(open(caps_json, 'r'))
    caption_embs = None
    if os.path.exists(emb_path):
        caption_embs = np.load(emb_path).astype('float32')
    return index, captions, caption_embs

def retrieve_by_image_embs(image_embs_np, index, caption_embs=None, captions=None, topk=5, normalize=True, device="cuda"):
    """
    image_embs_np: np.array [B, D]  (CLIP vision pooled vectors)
    index: faiss index built over caption_embs
    caption_embs: np.array [N, D]  (if None, only returns indices)
    captions: list[str] (optional)
    Returns:
      - retrieved_embs_t: torch.Tensor [B, topk, D] (or None if caption_embs None)
      - retrieved_texts: list[list[str]] per image (or None)
      - mask: torch.LongTensor [B, topk] with 1 valid, 0 pad
    """
    xq = image_embs_np.astype(np.float32)
    if normalize:
        faiss.normalize_L2(xq)
    D, I = index.search(xq, topk)  # D distances, I indices shape [B, topk]
    B = I.shape[0]

    retrieved_texts = None
    retrieved_embs = None
    if caption_embs is not None:
        # gather embeddings
        emb = caption_embs[I]  # [B, topk, D]
        # ensure float32
        emb = emb.astype('float32')
        # convert to torch tensor
        retrieved_embs = torch.from_numpy(emb).to(device)

    if captions is not None:
        retrieved_texts = []
        for irow in I:
            retrieved_texts.append([captions[i] for i in irow.tolist()])

    mask = torch.ones(B, topk, dtype=torch.long, device=device)
    return retrieved_embs, retrieved_texts, mask
 
def main(): 

    coco_data_path = '../data/dataset_coco.json' # path to Karpathy splits downloaded from Kaggle
    image_path = '../data/images/'
    
    print('Loading data')
    images, captions = load_coco_data(coco_data_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, feature_extractor = clip.load("RN50x64", device=device)

    print('Filtering captions')    
    xb_image_ids, captions = filter_captions(captions)

    print('Encoding captions')
    encoded_captions = encode_captions(captions, clip_model, device)
    
    print('Encoding images')
    xq_image_ids, encoded_images = encode_images(images, image_path, clip_model, feature_extractor, device)
    
    print('Retrieving neighbors')
    index, nns = get_nns(encoded_captions, encoded_images)
    retrieved_caps = filter_nns(nns, xb_image_ids, captions, xq_image_ids)

    print('Writing files')
    faiss.write_index(index, "../datastore/coco_index")
    json.dump(captions, open('../datastore/coco_index_captions.json', 'w'))

    json.dump(retrieved_caps, open('../data/retrieved_caps_resnet50x64.json', 'w'))

if __name__ == '__main__':
    main()




    

