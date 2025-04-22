# Semi Automatic Neuropath Analysis (SANA)

## About

SANA is a python-based package that was written by the [Penn Digital Neuropathology Lab](https://www.med.upenn.edu/digitalneuropathologylab/) to formalize our methods of IHC quantification. SANA includes functions which facilitate extracting pixel data from a Whole Slide Images (WSI), classifying pixels, and converting positive pixel masks to quantifications. 

## Installation
### Python
* Install >Python 3.9
### Dependencies
#### pip
`python3 -m pip install -r requirements.txt`
#### OpenSlide (may be required)
If the OpenSlide binaries are not found when running `import openslide`, following these [instructions](https://openslide.org/api/python/#installing)

## Getting Started

We provide several example [Jupyter](https://jupyter.org/) notebooks which contain example code blocks utilizing most of SANA's functionality.

* `examples/example0_prepare_images.ipynb` shows how to extract relevant ROI information from a WSI
* `examples/example1_process_images.ipynb` provides a sandbox for the preprocessing and pixel classification methods
* `examples/example2_normalize_cortex.ipynb` illustrates how to deform a curved section of cortex for more optimal quantification
* `examples/example3_quantification.ipynb` has examples of various quantification methods based on the positive pixel masks created by the previous notebooks

For more information, please refer to the [Documentation](https://penndigitalneuropathlab.github.io/sana/sana.html)

## Roadmap
* Automatic GM/WM segmentation
* Generic cell detection/segmentation
* Microglia detection/segmentation
* Structure Tensor Analysis
  
