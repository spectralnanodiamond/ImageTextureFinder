# ImageTextureFinder
A project to create an easy-to-use way of finding areas of common patterns and structures within an image. Should work on any image, designed for use on any biological images including DAPI, IMC and H&E.

A test image file is current located here:
https://zenodo.org/record/7821268#.ZDaHS3bMIuU

To install the environment, download the environment_lbp3a.yml file
then run 
`conda env create -f environment_lbp3a.yml`
then
`conda activate lbp3a`

Step 1: Run 2021-07-06_Local_binary_pattern_on_images.ipynb or 2022-10-28_Parallel_Local_binary_pattern_on_images_with_channels.ipynb

Use this script to run Local Binary Pattern over an image.
This outputs individual files for each condition where the x, y coordinates are for each patch and the z coordinate contains the LBP texture values


Step 2: 2022-10-28b_Creating_patch_LBP_signatures_with_channels_from_numpy_sources.ipynb

Combines the different LBP conditions into one file for easier analysis.

Step 3: 2022-11-01_UMAP_each_image_separately.ipynb

Run clustering and analysis in an unsupervised way.

