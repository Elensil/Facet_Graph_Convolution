
# Mesh Denoising with Facet Graph Convolutions

## Introduction
This repository provides all necessary code and information to reproduce the work presented in the following paper:
*Mesh Denoising with Facet Graph Convolutions*, TVCG (pending).
[Matthieu Armando](http://morpheo.inrialpes.fr/people/armando/), [Jean-Sébastien Franco](http://morpheo.inrialpes.fr/~franco/), [Edmond Boyer](http://morpheo.inrialpes.fr/people/Boyer/),
[[HAL]](https://hal.inria.fr/hal-02284101)

Please, bear in mind this is a research project, and we are currently working on making its installation and usage as user-friendly as possible. If you have issues, questions, or suggestions, do not hesitate to [contact us](#contact).
 
### Abstract

> We examine the problem of mesh denoising, which consists in removing noise from corrupted 3D meshes while preserving existing geometric features. Most mesh denoising methods require a lot of mesh-specific parameter fine-tuning, to account for specific features and noise types. In recent years, data-driven methods have demonstrated their robustness and effectiveness \wrt noise and feature properties on a wide variety of geometry and image problems. Most existing mesh denoising methods still use hand-crafted features, and locally denoise facets rather than examining the mesh globally. In this work, we propose the use of a fully end-to-end learning strategy based on graph convolutions, where meaningful features are learned directly by our network. Similar to most recent pipelines, given a noisy mesh, we first denoise face normals with our novel approach, then update vertices positions accordingly.  Our method performs significantly better than the current state-of-the-art learning-based methods. Additionally, we show that it can be trained on noisy data, without explicit correspondence between noisy and ground-truth facets. We also propose a multi-scale denoising strategy, better suited to correct noise with a low spatial frequency.


## Usage

- For a quick setup, download the pretrained model [here](https://drive.google.com/file/d/1mVqAnNFKQ-gdyNFGI1cj2eiYHVh2g3D6/view?usp=sharing), and extract it into the 'Networks' folder. Then, you can  change the paths in seetings.py (see below) and skip directly to inference. This model is trained on the synthetic dataset of [Wang et al.](https://wang-ps.github.io/denoising.html).
- Before you start, have a look have the settings.py file and set the path parameters. Make sure the specified folders exist. You can have a look at all the parameters, and you can tweak some of them once you are familiar with the whole program. You may have to if you have some memory issues.
- Raw mesh data need to be preprocessed before training. Run preprocess.py to preprocess your training dataset. This might take some time depending on your amount of data. By default, data will be saved to BINARY_DUMP_PATH parameter in settings. Alternatively, you can download preprocessed training data [here](https://drive.google.com/file/d/1jEMRQ9d0LTvB1HiX4XhrCHAiVX3cwmSt/view?usp=sharing) and validation data [here](https://drive.google.com/file/d/1Zu3GgvTruvwGKot8UXPeVZAuLWHpBlQs/view?usp=sharing), for the synthetic dataset. Extract them into the 'Preprocessed_Data' folder.
- Once you have generated binary dump files of your preprocessed training set, run train.py to train a model.
- For inference, run infer.py --input_dir /my/input/dir/


## Tips

- If you run out of memory at runtime, try to lower the value for MAX_PATCH_SIZE in settings.py. For training, you will have to re-generate training binaries with the new parameter, for it to take effect.
- By default, the program will run without the two extensions presented at the end of the paper, i.e. with a loss on face normals only, and without multi-scale normal estimation. You can change this setting with the INCLUDE_VERTICES parameter. If so, please be aware that:
	- You will need to run the preprocessing step with this setting. (Data generated in this manner can be used for both modes, but they will take longer to generate and take up more space, which is why a lighter version is computed by default).
	- Training will be longer, and you might need more data to get good results
	- Results might look less smooth, and it might be tricky to get a better accuracy, depending on your data. Some potential leeway is still hard-coded. Do not hesitate to contact us for more in-depth tests.
- Some parameters can be overriden by command line arguments (e.g. --network_path /my/path/ or --results_path /my/path/). See settings file for more details.
- If no input directory is provided for inference, the network will run on the validation data by default.


## Data
We use the data from "Mesh Deoising via Cascaded Normal Regression".
See their [github page](https://wang-ps.github.io/denoising.html) for access to their data.

## Citation
If you use this code, please cite the following:
```
@ARTICLE{armando21,  
  title     = {Mesh Denoising with Facet Graph Convolutions},  
  author    = {M. {Armando} and J.S. {Franco} and E. {Boyer}},  
  journal = {IEEE Transactions on Visualization and Computer Graphics},
  year      = {2021},
  volume={},
  number={},
  pages={1-1}
}
```

## License
Please check the [license terms](https://gitlab.inria.fr/marmando/deep-mesh-denoising/blob/master/LICENSE.md) before downloading and/or using the code.


## Contact
[Matthieu Armando](http://morpheo.inrialpes.fr/people/armando/)
 - INRIA Grenoble Rhône-Alpes
 - 655, avenue de l’Europe, Montbonnot
 - 38334 Saint Ismier, France
 - Email: [matthieu.armando@inria.fr](mailto:matthieu.armando@inria.fr)

## Acknowledgements

The GCN code was originally built on top of the [FeaStNet](https://github.com/nitika-verma/FeaStNet) project. As such, some bits were written by [Nitika Verma](https://nitika-verma.github.io/).
The graph coarsening method we used is the one presented in "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering", and we make use of [their code](https://github.com/mdeff/cnn_graph) (with minor corrections).

## Misc
- Some visualization tools used in the paper (heatmaps of angular error, coloring faces by predicted normals, etc.) have not been included. They might be added in the future.