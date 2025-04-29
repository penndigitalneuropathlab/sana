# PDNL Semi Automatic Neuropath Analysis (PDNL-SANA)

## About

PDNL-SANA is a python-based package written by the [Penn Digital Neuropathology Lab](https://www.med.upenn.edu/digitalneuropathologylab/) to formalize our methods of IHC quantification. PDNL-SANA includes functions which facilitate extracting pixel data from a Whole Slide Images (WSI), classifying pixels, and converting positive pixel masks to quantifications. 

## Requirements

python3.9 or greater

## Installation

`python3 -m pip install pdnl_sana`

## Getting Started

We provide several example [Jupyter](https://jupyter.org/) notebooks which contain example code blocks utilizing most of SANA's functionality.

* `docs/source/examples/example0_prepare_images.ipynb` shows how to extract relevant ROI information from a WSI
* `docs/source/examples/example1_process_images.ipynb` provides a sandbox for the preprocessing and pixel classification methods
* `docs/source/examples/example2_normalize_cortex.ipynb` illustrates how to deform a curved section of cortex for more optimal quantification
* `docs/source/examples/example3_quantification.ipynb` has examples of various quantification methods based on the positive pixel masks created by the previous notebooks

For more information, please refer to the [Documentation](https://pdnl-sana.readthedocs.io/en/latest/)

## Roadmap
* GPU Acceleration 
* Automatic GM/WM segmentation
* Generic cell detection/segmentation
* Microglia detection/segmentation
* Structure Tensor Analysis

