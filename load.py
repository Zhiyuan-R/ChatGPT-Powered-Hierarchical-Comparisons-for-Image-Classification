import json
import numpy as np
import torch
from torch.nn import functional as F

from descriptor_strings import *  # label_to_classname, wordify, modify_descriptor,
import pathlib

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageNet, ImageFolder
from imagenetv2_pytorch import ImageNetV2Dataset as ImageNetV2
from datasets import _transform, CUBDataset, _transform_color, _transform_edge, _transform_lp
from collections import OrderedDict
import clip

from loading_helpers import *


hparams = {}
# hyperparameters

hparams['model_size'] = "ViT-B/32" 
# Options:
# ['RN50',
#  'RN101',
#  'RN50x4',
#  'RN50x16',
#  'RN50x64',
#  'ViT-B/32',
#  'ViT-B/16',
#  'ViT-L/14',
#  'ViT-L/14@336px']
hparams['dataset'] = 'cub'
# hparams['dataset'] = 'imagenet'
# hparams['dataset'] = 'food'
# hparams['dataset'] = 'pet'
# hparams['dataset'] = 'place'
# hparams['dataset'] = 'texture'

hparams['batch_size'] = 96*10
hparams['device'] = "cuda" if torch.cuda.is_available() else "cpu"
hparams['category_name_inclusion'] = 'prepend' #'append' 'prepend'

hparams['apply_descriptor_modification'] = True

hparams['verbose'] = False
hparams['image_size'] = 224
if hparams['model_size'] == 'ViT-L/14@336px' and hparams['image_size'] != 336:
    print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 336.')
    hparams['image_size'] = 336
elif hparams['model_size'] == 'RN50x4' and hparams['image_size'] != 288:
    print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 288.')
    hparams['image_size'] = 288
elif hparams['model_size'] == 'RN50x16' and hparams['image_size'] != 384:
    print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 288.')
    hparams['image_size'] = 384
elif hparams['model_size'] == 'RN50x64' and hparams['image_size'] != 448:
    print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 288.')
    hparams['image_size'] = 448

hparams['before_text'] = ""
hparams['label_before_text'] = ""
hparams['between_text'] = ', '
hparams['after_text'] = ''
hparams['unmodify'] = True
hparams['label_after_text'] = ''
hparams['seed'] = 1

# TODO: fix this... defining global variable to be edited in a function, bad practice
# unmodify_dict = {}

# classes_to_load = openai_imagenet_classes
hparams['descriptor_fname'] = None

IMAGENET_DIR = '' # REPLACE THIS WITH YOUR OWN PATH
CUB_DIR = 'data/CUB_200_2011' # REPLACE THIS WITH YOUR ViT-L/14OWN PATH
FOOD_DIR = '' # REPLACE THIS WITH YOUR OWN PATH
PET_DIR = '' # REPLACE THIS WITH YOUR OWN PATH
PLACE_DIR = '' # REPLACE THIS WITH YOUR OWN PATH
TEXTURE_DIR = '' # REPLACE THIS WITH YOUR OWN PATH

# PyTorch datasets
tfms = _transform(hparams['image_size'])

if hparams['dataset'] == 'imagenet':
    if hparams['dataset'] == 'imagenet':
        dsclass = ImageNet        
        hparams['data_dir'] = pathlib.Path(IMAGENET_DIR)
        dataset = dsclass(hparams['data_dir'], split='val', transform=tfms)
        classes_to_load = None
    
        if hparams['descriptor_fname'] is None:
             hparams['descriptor_fname'] = '' # filename
        hparams['after_text'] = hparams['label_after_text'] = '.'
        
    elif hparams['dataset'] == 'imagenetv2':
        hparams['data_dir'] = pathlib.Path(IMAGENETV2_DIR)
        dataset = ImageNetV2(location=hparams['data_dir'], transform=tfms)
        classes_to_load = openai_imagenet_classes
        hparams['descriptor_fname'] = 'descriptors_imagenet'
        dataset.classes = classes_to_load

elif hparams['dataset'] == 'cub':
    # load CUB dataset
    hparams['data_dir'] = pathlib.Path(CUB_DIR)
    dataset = CUBDataset(hparams['data_dir'], train=False, transform=tfms)
    classes_to_load = None #dataset.classes
    hparams['descriptor_fname'] = 'cub-10-2.json'

elif hparams['dataset'] == 'food':
    hparams['data_dir'] = pathlib.Path(FOOD_DIR)
    dataset = Food101(hparams['data_dir'])
    classes_to_load = None
    hparams['descriptor_fname'] = '' # filename
    
elif hparams['dataset'] == 'pet':
    hparams['data_dir'] = pathlib.Path(PET_DIR)
    dataset = OxfordPet(hparams['data_dir'])
    classes_to_load = None
    hparams['descriptor_fname'] = '' # filename
    
elif hparams['dataset'] == 'place':
    hparams['data_dir'] = pathlib.Path(PLACE_DIR)
    dataset = Places365(hparams['data_dir'])
    classes_to_load = None
    hparams['descriptor_fname'] = '' # filename
    
elif hparams['dataset'] == 'texture':
    hparams['data_dir'] = pathlib.Path(TEXTURE_DIR)
    dataset = DTD(hparams['data_dir'])
    classes_to_load = None
    hparams['descriptor_fname'] = '' # filename

hparams['descriptor_fname'] = './descriptions/' + hparams['descriptor_fname']
    

print("Creating descriptors...")



tree_dict = load_json(hparams['descriptor_fname'])
label_to_classname = list(tree_dict.keys())


n_classes = len(list(tree_dict.keys()))

def compute_tree_encodings(model, tokenizer=clip.tokenize):
    description_encodings = OrderedDict()
    for k, _ in tree_dict.items():
        description_encodings[k] = [] 
    for k, v in tree_dict.items():
        for x in v:
            tokens = tokenizer(x).to(hparams['device'])
            description_encodings[k].append(F.normalize(model.encode_text(tokens)))
    return description_encodings

def compute_description_encodings(model):
    description_encodings = OrderedDict()
    for k, v in gpt_descriptions.items():
        tokens = clip.tokenize(v).to(hparams['device'])
        description_encodings[k] = F.normalize(model.encode_text(tokens))
    return description_encodings

def compute_label_encodings(model, tokenizer=clip.tokenize):
    label_encodings = F.normalize(model.encode_text(tokenizer([hparams['label_before_text'] + wordify(l) + hparams['label_after_text'] for l in label_to_classname]).to(hparams['device'])))
    return label_encodings

def compute_attributes_encodings(model):
    attr_encodings = F.normalize(model.encode_text(tokenizer([hparams['label_before_text'] + wordify(l) + hparams['label_after_text'] for l in question_list]).to(hparams['device'])))
    return attr_encodings

def aggregate_similarity(similarity_matrix_chunk, aggregation_method='mean', similarity_labels = None):
    if aggregation_method == 'max': return similarity_matrix_chunk.max(dim=1)[0]
    elif aggregation_method == 'sum': return similarity_matrix_chunk.sum(dim=1)
    elif aggregation_method == 'mean': return similarity_matrix_chunk.mean(dim=1)
    elif aggregation_method == 'cut':
        similarity_matrix_total = torch.cat((similarity_matrix_chunk, similarity_labels[:, None]), dim=1)
        
        similarity_matrix_total_zeromean = similarity_matrix_total - similarity_labels[:, None]
        mask = ~(similarity_matrix_total_zeromean < 0)
        non_neg = mask.sum(dim=1)
        
        similarity_matrix_total[similarity_matrix_total<similarity_labels[:, None]] = 0
        similarity_matrix_total = similarity_matrix_total.sum(dim=1) / non_neg
        
        return similarity_matrix_total

    else: raise ValueError("Unknown aggregate_similarity")

def show_from_indices(indices, images, labels=None, predictions=None, predictions2 = None, n=None, image_description_similarity=None, image_labels_similarity=None):
    if indices is None or (len(indices) == 0):
        print("No indices provided")
        return
    
    if n is not None:
        indices = indices[:n]
    
    for index in indices:
        show_single_image(images[index])
        print(f"Index: {index}")
        if labels is not None:
            true_label = labels[index]
            true_label_name = label_to_classname[true_label]
            print(f"True label: {true_label_name}")
        if predictions is not None:
            predicted_label = predictions[index]
            predicted_label_name = label_to_classname[predicted_label]
            print(f"Predicted label (ours): {predicted_label_name}")
        if predictions2 is not None:
            predicted_label2 = predictions2[index]
            predicted_label_name2 = label_to_classname[predicted_label2]
            print(f"Predicted label 2 (CLIP): {predicted_label_name2}")
        
        print("\n")
        
        if image_labels_similarity is not None:
            if labels is not None:
                print(f"Total similarity to {true_label_name} (true label) labels: {image_labels_similarity[index][true_label].item()}")
            if predictions is not None:
                if labels is not None and true_label_name == predicted_label_name: 
                    print("Predicted label (ours) matches true label")
                else: 
                    print(f"Total similarity to {predicted_label_name} (predicted label) labels: {image_labels_similarity[index][predicted_label].item()}")
            if predictions2 is not None:
                if labels is not None and true_label_name == predicted_label_name2: 
                    print("Predicted label 2 (CLIP) matches true label")
                elif predictions is not None and predicted_label_name == predicted_label_name2: 
                    print("Predicted label 2 (CLIP) matches predicted label 1")
                else: 
                    print(f"Total similarity to {predicted_label_name2} (predicted label 2) labels: {image_labels_similarity[index][predicted_label2].item()}")
        
            print("\n")
        
        if image_description_similarity is not None:
            if labels is not None:
                print_descriptor_similarity(image_description_similarity, index, true_label, true_label_name, "true")
                print("\n")
            if predictions is not None:
                if labels is not None and true_label_name == predicted_label_name:
                    print("Predicted label (ours) same as true label")
                    # continue
                else:
                    print_descriptor_similarity(image_description_similarity, index, predicted_label, predicted_label_name, "descriptor")
                print("\n")
            if predictions2 is not None:
                if labels is not None and true_label_name == predicted_label_name2:
                    print("Predicted label 2 (CLIP) same as true label")
                    # continue
                elif predictions is not None and predicted_label_name == predicted_label_name2: 
                    print("Predicted label 2 (CLIP) matches predicted label 1")
                else:
                    print_descriptor_similarity(image_description_similarity, index, predicted_label2, predicted_label_name2, "CLIP")
            print("\n")

def print_descriptor_similarity(image_description_similarity, index, label, label_name, label_type="provided"):
    # print(f"Total similarity to {label_name} ({label_type} label) descriptors: {aggregate_similarity(image_description_similarity[label][index].unsqueeze(0)).item()}")
    print(f"Total similarity to {label_name} ({label_type} label) descriptors:")
    print(f"Average:\t\t{100.*aggregate_similarity(image_description_similarity[label][index].unsqueeze(0)).item()}")
    label_descriptors = gpt_descriptions[label_name]
    for k, v in sorted(zip(label_descriptors, image_description_similarity[label][index]), key = lambda x: x[1], reverse=True):
        k = unmodify_dict[label_name][k]
        # print("\t" + f"matched \"{k}\" with score: {v}")
        print(f"{k}\t{100.*v}")
        
def print_max_descriptor_similarity(image_description_similarity, index, label, label_name):
    max_similarity, argmax = image_description_similarity[label][index].max(dim=0)
    label_descriptors = gpt_descriptions[label_name]
    print(f"I saw a {label_name} because I saw {unmodify_dict[label_name][label_descriptors[argmax.item()]]} with score: {max_similarity.item()}")
    
def show_misclassified_images(images, labels, predictions, n=None, 
                              image_description_similarity=None, 
                              image_labels_similarity=None,
                              true_label_to_consider: int = None, 
                              predicted_label_to_consider: int = None):
    misclassified_indices = yield_misclassified_indices(images, labels=labels, predictions=predictions, true_label_to_consider=true_label_to_consider, predicted_label_to_consider=predicted_label_to_consider)
    if misclassified_indices is None: return
    show_from_indices(misclassified_indices, images, labels, predictions, 
                      n=n,
                      image_description_similarity=image_description_similarity, 
                      image_labels_similarity=image_labels_similarity)

def yield_misclassified_indices(images, labels, predictions, true_label_to_consider=None, predicted_label_to_consider=None):
    misclassified_indicators = (predictions.cpu() != labels.cpu())
    if true_label_to_consider is not None:
        misclassified_indicators = misclassified_indicators & (labels.cpu() == true_label_to_consider)
    if predicted_label_to_consider is not None:
        misclassified_indicators = misclassified_indicators & (predictions.cpu() == predicted_label_to_consider)
        
    if misclassified_indicators.sum() == 0:
        output_string = 'No misclassified images found'
        if true_label_to_consider is not None:
            output_string += f' with true label {label_to_classname[true_label_to_consider]}'
        if predicted_label_to_consider is not None:
            output_string += f' with predicted label {label_to_classname[predicted_label_to_consider]}'
        print(output_string + '.')
            
        return
    
    misclassified_indices = torch.arange(images.shape[0])[misclassified_indicators]
    return misclassified_indices


from PIL import Image
def predict_and_show_explanations(images, model, labels=None, description_encodings=None, label_encodings=None, device=None):
    if type(images) == Image:
        images = tfms(images)
        
    if images.device != device:
        images = images.to(device)
        labels = labels.to(device)

    image_encodings = model.encode_image(images)
    image_encodings = F.normalize(image_encodings)
    
    
    
    image_labels_similarity = image_encodings @ label_encodings.T
    clip_predictions = image_labels_similarity.argmax(dim=1)
    
    n_classes = len(description_encodings)
    image_description_similarity = [None]*n_classes
    image_description_similarity_cumulative = [None]*n_classes
    for i, (k, v) in enumerate(description_encodings.items()): # You can also vectorize this; it wasn't much faster for me
        
        
        dot_product_matrix = image_encodings @ v.T
        
        image_description_similarity[i] = dot_product_matrix
        image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i])
        
        
    # create tensor of similarity means
    cumulative_tensor = torch.stack(image_description_similarity_cumulative,dim=1)
        
    
    descr_predictions = cumulative_tensor.argmax(dim=1)
    
    
    show_from_indices(torch.arange(images.shape[0]), images, labels, descr_predictions, clip_predictions, image_description_similarity=image_description_similarity, image_labels_similarity=image_labels_similarity)