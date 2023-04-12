# ImageTextureFinder
A project to create an easy-to-use way of finding areas of common patterns and structures within an image

A test image file is current located here:
https://zenodo.org/record/7821268#.ZDaHS3bMIuU

To install the environment, use:

mamba create --name  lbp1 -c conda-forge -c bioconda -c ohsu-comp-bio -c anaconda -c menpo -c intel scanpy numba hdbscan umap-learn python-igraph leidenalg ipython anndata squidpy pandas matplotlib seaborn jupyter tqdm vispy ipywidgets dask pandas opencv scikit-image napari dask-image scipy seaborn scikit-learn xlrd openpyxl imagecodecs-lite colour h5py xlwt nmslib matplotlib-scalebar openslide-python ez_setup moviepy mahotas cython pip

pip install --upgrade fastremap

Step 1: 2022-10-28_Parallel_Local_binary_pattern_on_images_with_channels.ipynb

Use this script to run Local Binary Pattern over an image.
This outputs individual files for each condition where the x, y coordinates are for each patch and the z coordinate contains the LBP texture values


Step 2: 2022-10-28b_Creating_patch_LBP_signatures_with_channels_from_numpy_sources.ipynb

Combines the different LBP conditions into one file for easier analysis.

Step 3: 2022-11-01_UMAP_each_image_separately.ipynb

Run clustering and analysis.

