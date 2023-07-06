# ImageTextureFinder
A project to create an easy-to-use way of finding areas of common patterns and structures within an image. Should work on any image, designed for use on any biological images including DAPI, IMC and H&E.

Ben's attempt to make a parallel version using joblib is in the other branch 'B_parallel'. 

A test image file is current located here:
https://zenodo.org/record/7821268#.ZDaHS3bMIuU

To install the environment, download the environment_lbp3c.yml file or the environment_lbp3b_windows.yml file
then run 
`conda env create -f environment_lbp3c.yml` or on windows `conda env create -f environment_lbp3b_windows.yml`
then
`conda activate lbp3c`
or
`conda activate lbp3b_windows`

(Side note: if you would like to use ndpi files as input, then run `apt install openslide-tools` before creating the environments above)

Step 1: Run `2021-07-06_Local_binary_pattern_on_images.ipynb`

Use this script to run Local Binary Pattern over an image.
This outputs individual files for each condition where the x, y coordinates are for each patch and the intensity contains the LBP texture value. Multiple images are created to hold the LBP information from different radii.


Step 2: Run `2021-07-28_Creating_patch_LBP_signatures_flexible_LBP_sources.ipynb`

Combines the different LBP conditions into one file for easier analysis. 

Step 3: Run `2022-11-01_UMAP_each_image_separately.ipynb` or `2022-11-10_UMAP_and_clustering_all_12_images` 
Either of these can be run independently of which options you chose above
Run clustering and analysis in an unsupervized way.

Step 4 (optional): If you ran `2022-11-10_UMAP_and_clustering_all_12_images` then you will have output some clusters that can be displayed on an image that can be viewed in napari with this script: `2022-11-10d_View_cluster_masks_in_napari.ipynb`
