import os
import openai
import sys
from load import *
import torch
from tqdm import tqdm
import itertools
from descriptor_strings import stringtolist
import json
import clip
import numpy as np
import math
from collections import defaultdict
import time

openai.api_key = "your key"

### hyperparameter 
# start

# --variable--
device = "cuda" if torch.cuda.is_available() else "cpu"
label_to_classname = label_to_classname
num_group_div = 6 # modify this according to different dataset
th = 3 # modify this according to different dataset
before_text = ''
between_text = ', '
after_text = ''

# --storage--
descriptors = defaultdict(list)

# end

### generating the base
# start 

def generate_prompt(category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: What are useful visual features for distinguishing a lemur in a photo?
A: There are several useful visual features to tell there is a lemur in a photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful visual features for distinguishing a television in a photo?
A: There are several useful visual features to tell there is a television in a photo:
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control

Q: What are useful features for distinguishing a {category_name} in a photo?
A: There are several useful visual features to tell there is a {category_name} in a photo:
-
"""

def wordify(string):
    word = string.replace('_', ' ')
    return word

def make_descriptor_sentence(descriptor):
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"
    
def modify_descriptor(descriptor, apply_changes):
    if apply_changes:
        return make_descriptor_sentence(descriptor)
    return descriptor

prompt_list =  [generate_prompt(category.replace('_', ' ')) for category in label_to_classname]

for idx, prompt in tqdm(enumerate(prompt_list), total=len(prompt_list)):
    
    while(True):
        try:
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "user", "content": prompt}
            ]
        )
            break
        except:
            time.sleep(3)
    
    string = completion["choices"][0]["message"]["content"]

    string = stringtolist(string)
    
    key = label_to_classname[idx]
        
    word_to_add = wordify(key)
    
    build_descriptor_string = lambda item: f"{before_text}{word_to_add}{between_text}{modify_descriptor(item, True)}{after_text}"

    descriptors[key].append([build_descriptor_string(s) for s in string])
    
### grouping and add specific description
# start

model, preprocess = clip.load(hparams['model_size'], device=device, jit=False, download_root='./')
model.eval()
model.requires_grad_(False)

def k_means(data, k, max_iters=300):
    """
    Runs the k-means algorithm on the given data.

    Args:
        data (torch.Tensor): The data to cluster, of shape (N, D).
        k (int): The number of clusters to form.
        max_iters (int): The maximum number of iterations to run the algorithm for.

    Returns:
        A tuple containing:
        - cluster_centers (torch.Tensor): The centers of the clusters, of shape (k, D).
        - cluster_assignments (torch.Tensor): The cluster assignments for each data point, of shape (N,).
    """
    # Initialize cluster centers randomly
    # Initialize cluster centers randomly
    np.random.seed(42)
    cluster_centers = data[np.random.choice(data.shape[0], k, replace=False)]
    cluster_assignments = None

    # Run the algorithm for a fixed number of iterations
    for i in range(max_iters):
        # Compute distances between data and cluster centers using broadcasting
        distances = torch.norm(data[:, None, :] - cluster_centers[None, :, :], dim=-1)
        # Assign each data point to the nearest cluster center
        cluster_assignments = torch.argmin(distances, dim=1)

        # Update the cluster centers based on the mean of the assigned points
        for j in range(k):
            mask = cluster_assignments == j
            if mask.any():
                cluster_centers[j] = data[mask].mean(dim=0)

    return cluster_centers, cluster_assignments

def generate_prompt_summary(category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q:summarize the following categories with one sentence: Salmon, Goldfish, Piranha, Zebra Shark, Whale Shark, Snapper, Swordfish, Bass, Trout?
A:this is a dataset of various fishes

Q:summarize the following categories with one sentence: Smartphone, Laptop, Piranha, Scanner, Refrigerator, Tiger, Bluetooth Speaker, Projector, Printer?
A:this dataset includes different electronic devices

Q:summarize the following categories with one sentence: Scott Oriole, Baird Sparrow, Black-throated Sparrow, Chipping Sparrow, House Sparrow, Grasshopper Sparrow
A:most categories in this dataset are sparrow

Q: summarize the following categories with one sentence: {category_name}?
A: 
"""

def generate_prompt_given_overall_feature(category_name: str, over_all: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: What are useful visual features for distinguishing a Clay Colored Sparrow in a photo in a dataset: This dataset consists of various sparrows?
A: There are several useful visual features to tell there is a Clay Colored Sparrow in a photo:
- a distinct pale crown stripe or central crown patch
- a dark eyeline and a pale stripe above the eye
- brownish-gray upperparts
- conical-shaped bill

Q: What are useful visual features for distinguishing a Zebra Shark in a photo in a dataset: Most categories in this dataset are sharks?
A: There are several useful visual features to tell there is a Zebra Shark in a photo:
- prominent dark vertical stripes or bands
- a sleek and slender body with a long, flattened snout and a distinctive appearance
- a tan or light brown base color on their body
- a long, slender tail with a pattern of dark spots and bands that extend to the tail fin
- dark edges of both dorsal fins

Q: What are useful features for distinguishing a {category_name} in a photo: {over_all}?
A: There are several useful visual features to tell there is a {category_name} in a photo:
-
"""

def generate_description_overall(categories_group):
    string = ', '.join(categories_group)
    prompt = generate_prompt_summary(string)

    while(True):
        try:
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "user", "content": prompt}
            ]
        )
            break
        except:
            time.sleep(3)    
    overall_feature = completion["choices"][0]["message"]["content"]
    
    print("overall_feature", overall_feature)
    print("they are describing", prompt)
    
    prompt_list =  [generate_prompt_given_overall_feature(category.replace('_', ' '), overall_feature) for category in categories_group]
    
    for idx, prompt in tqdm(enumerate(prompt_list), total=len(prompt_list)):
        while(True):
            try:
                completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                {"role": "user", "content": prompt}
                ]
            )
                break
            except:
                time.sleep(3)
        
        string = completion["choices"][0]["message"]["content"]
        string = stringtolist(string)
        
        key = categories_group[idx]
        
        word_to_add = wordify(key)
        
        build_descriptor_string = lambda item: f"{before_text}{word_to_add}{between_text}{modify_descriptor(item, True)}{after_text}"

        descriptors[key].append([build_descriptor_string(s) for s in string])
      
def generate_prompt_compare(categories_group: str, to_compare: str):
    return f"""Q: What are useful visual features for distinguishing Hooded Oriole from Scott Oriole, Baltimore Oriole in a photo
A: There are several useful visual features to tell there is a Hooded Oriole in a photo:
- distinctive bright orange or yellow and black coloration
- orange or yellow body and underparts
- noticeably curved downwards bill
- a black bib or "hood" that extends up over the head and down the back

Q: What are useful visual features for distinguishing a smartphone from television, laptop, scanner, printer in a photo?
A: There are several useful visual features to tell there is a smartphone in a photo:
- rectangular and much thinner shape
- a touchscreen, lacking the buttons and dials
- manufacturer's logo or name visible on the front or back of the device
- one or more visible camera lenses on the back

Q: What are useful features for distinguishing a {categories_group} from {to_compare} in a photo?
A: There are several useful visual features to tell there is a {categories_group} in a photo:
-
"""
        
def generate_description_compare(categories_group):

    for x in categories_group:
        subtracted_list = [y for y in categories_group if y != x]
        string = ', '.join(subtracted_list)
        prompt = generate_prompt_compare(categories_group, string)
    
        while(True):
            try:
                completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                {"role": "user", "content": prompt}
                ]
            )
                break
            except:
                time.sleep(3)
        
        res = completion["choices"][0]["message"]["content"]
        res = stringtolist(res)
        
        key = x
        
        word_to_add = wordify(key)
        
        build_descriptor_string = lambda item: f"{before_text}{word_to_add}{between_text}{modify_descriptor(item, True)}{after_text}"

        descriptors[key].append([build_descriptor_string(s) for s in res])



def build_tree_in_loop(class_names):

    description_encodings = OrderedDict()
    for k, v in descriptors.items():
        if k in class_names:
            tokens = clip.tokenize(v[-1]).to(device)
            description_encodings[k] = F.normalize(model.encode_text(tokens))
            
    text_avg_emb = [None]*len(description_encodings)
    for i, (k,v) in enumerate(description_encodings.items()):
        text_avg_emb[i] = v.mean(dim=0)
    try:
        text_avg_emb = torch.stack(text_avg_emb, dim=0)
    except:
        import pdb
        pdb.set_trace()
    
    num_group = int(math.ceil((len(class_names) / num_group_div)))
    if num_group<= 1:
        num_group=2
    
    _, cluster_assignments = k_means(text_avg_emb, num_group)
    
    label_to_classname_np = np.array(class_names)

    for group_idx in range(num_group):
        tmp_index = torch.where(cluster_assignments == group_idx)[0]
        
        categories_group = label_to_classname_np[tmp_index.cpu()]
        if isinstance(categories_group, np.ndarray):
            categories_group = categories_group.tolist()
        if not isinstance(categories_group, list):
            categories_group = [categories_group]
        
        if len(categories_group) <= th and len(categories_group)>=2:
            print("direct comparison")
            print(categories_group)
            generate_description_compare(categories_group)
        elif len(categories_group) <= 1:
            print("lonely!")
            print(categories_group)
        else:
            print("summary")
            generate_description_overall(categories_group)
            build_tree_in_loop(categories_group)
            
            
build_tree_in_loop(label_to_classname)
    
### debug
# start

for k, v in descriptors.items():
    print(k, len(v))

# end
    
with open("output.json", "w") as file:
    json.dump(descriptors, file)
    
