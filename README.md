# ImageTextureFinder
A project to create an easy-to-use way of finding areas of common patterns and structures within an image. Should work on any image, designed for use on any biological images including DAPI, IMC and H&E.

A test image file is current located here:
https://zenodo.org/record/7821268#.ZDaHS3bMIuU

To install the environment, download the environment_lbp3c.yml file or the environment_lbp3b_windows.yml file
then run 
`conda env create -f environment_lbp3c.yml` or on windows `conda env create -f environment_lbp3b_windows.yml`
then
`conda activate lbp3c`
or
`conda activate lbp3b_windows`

Step 1: Run `2021-07-06_Local_binary_pattern_on_images.ipynb` or `2022-10-28_Parallel_Local_binary_pattern_on_images_with_channels.ipynb`

Use this script to run Local Binary Pattern over an image.
This outputs individual files for each condition where the x, y coordinates are for each patch and the z coordinate contains the LBP texture values


Step 2: Run `2021-07-28_Creating_patch_LBP_signatures_flexible_LBP_sources.ipynb` or `2022-10-28b_Creating_patch_LBP_signatures_with_channels_from_numpy_sources.ipynb`

Combines the different LBP conditions into one file for easier analysis. Notebook used in this step must be the one close in date to the notebook used in step 1.

Step 3: Run `2022-11-01_UMAP_each_image_separately.ipynb` or `2022-11-10_UMAP_and_clustering_all_12_images` 
Either of these can be run independently of which options you chose above
Run clustering and analysis in an unsupervised way.

Step 4 (optional): If you ran `2022-11-10_UMAP_and_clustering_all_12_images` then you will have output some clusters that can be displayed on an image that can be viewed in napari with this script: `2022-11-10d_View_cluster_masks_in_napari.ipynb`
