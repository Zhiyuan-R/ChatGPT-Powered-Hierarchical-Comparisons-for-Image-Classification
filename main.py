from load import *
from loading_helpers import *
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import clip
import open_clip


seed_everything(hparams['seed'])

bs = hparams['batch_size']
dataloader = DataLoader(dataset, bs, shuffle=False, num_workers=16, pin_memory=True)

print("Loading model...")

device = torch.device(hparams['device'])
## if load model in CLIP (modify model size in the load.py)
model, preprocess = clip.load(hparams['model_size'], device=device, jit=False, download_root='./')
##  or load openclip model (comment the last line and choose the model and pretrained_dataset for openclip's models)
# model_name = 'ViT-bigG-14'
# pretrained_dataset = 'laion2b_s39b_b160k'
# model_name = 'ViT-L-14'
# pretrained_dataset = 'laion400m_e32'
# model_name = 'ViT-B-16'
# pretrained_dataset = 'laion400m_e32'

# model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_dataset, cache_dir = './')
# model.to(device)

model.eval()
model.requires_grad_(False)

print("Encoding descriptions...")


try:
    tree_encodings = compute_tree_encodings(model)
    label_encodings = compute_label_encodings(model)
except:
    tree_encodings = compute_tree_encodings(model, open_clip.get_tokenizer(model_name))
    label_encodings = compute_label_encodings(model, open_clip.get_tokenizer(model_name))


sample_num = 0
clip_correct_num = 0
descr_correct_num = 0
tree_correct_num = 0
eta = 0

best = 0

lamda = 0.5

predict_clip_list = []
predict_ours_list = []
label_list = []

for batch_number, batch in enumerate(tqdm(dataloader)):

    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)
    sample_num += labels.shape[0]
    label_list.append(labels)

    # ---- CLIP AREA----- 
    # START
    image_encodings = model.encode_image(images)
    image_encodings = F.normalize(image_encodings)
    
    image_clip_similarity = image_encodings @ label_encodings.T
    clip_predictions = image_clip_similarity.argmax(dim=1)

    clip_correct_num += (clip_predictions== labels).sum()
    
    # END
    predict_clip_list.append(clip_predictions)
    
    # ---- TREE AREA----- 
    # START
    
    image_tree_similarity_cumulative = [None]*n_classes
    
    
    for i, (k, v) in enumerate(tree_encodings.items()):
        dot_product_matrix_base = (image_encodings @ v[0].T).mean(dim=1)
        if len(v) > 1:
            
            score = torch.stack([(image_encodings @ f.T).mean(dim=1) for f in v[1:]], dim=1)
            diffs = score[:, 1:] - score[:, :-1]
            padded_diffs = F.pad(diffs, (1, 0, 0, 0), value=1)
            mask = padded_diffs > eta
            
            
            first_false = (mask == False).cumsum(dim=1) >= 1

            # Set values to False after the first False in each row
            mask[first_false] = False

            dot_product_matrix_comp = (score * mask).sum(dim=1) / mask.sum(dim=1)
            image_tree_similarity_cumulative[i] = lamda * dot_product_matrix_base + (1-lamda) * dot_product_matrix_comp
            
        else:
            image_tree_similarity_cumulative[i] = dot_product_matrix_base
        
    
    cumulative_tensor_tree = torch.stack(image_tree_similarity_cumulative, dim=1)
    tree_predictions = cumulative_tensor_tree.argmax(dim=1)
    
    tree_correct_num += (tree_predictions== labels).sum()
    
    # END
    predict_ours_list.append(tree_predictions)
    
print('CLIP standard accuracy: {}'.format(clip_correct_num/sample_num))
print('Tree-based accuracy: {}'.format(tree_correct_num/sample_num))

