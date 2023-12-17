# :robot: :robot: :robot: ChatGPT-Powered Hierarchical Comparisons for Image Classification
Source Code for Neurips 2023 Publication: &lt;ChatGPT-Powered Hierarchical Comparisons for Image Classification>

Paper Link: https://cvlab.cse.msu.edu/pdfs/ren_su_liu_neurips2023.pdf

## Get Started!
Packages:
* python=3.9
* pytorch
* matplotlib
* CLIP
* ImageNetV2_pytorch
* scikit-learn
* seaborn
* open_clip

Put each dataset under "./data/".

## Generate the description tree!
```python
python generate_tree.py
```

(note: add your own openai_key; modify "num_group_div"; modify "th")

## Do inference
```python
python main.py
```
