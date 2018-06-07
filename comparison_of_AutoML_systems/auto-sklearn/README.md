# Usage

This folder contains files used to generate plots showing the change of performance of auto-sklearn with runtime budget, as compared to oboe. The collection of algorithms we run is the same as what we set for oboe.

After specifying the dataset indices to run auto-sklearn on in `autosklearn_OpenML.sh`, as well as the directory containing pre-processed OpenML datasets,
```
bash autosklearn_OpenML.sh
```
will generate `.csv` files containing performance of auto-sklearn on this collection of datasets and output them into a separate folder, whose path is specified in `autosklearn_results.py`.

Similarly,
```
bash autosklearn_UCI.sh
```
will generate results for pre-processed UCI datasets, or for any datasets with filenames as dataset names.

We spotted some unknown execution errors and larger classification errors when running auto-sklearn on a batch of datasets in parallel, thus we sequentially ran auto-sklearn on datasets to maximize the performance of auto-sklearn.
