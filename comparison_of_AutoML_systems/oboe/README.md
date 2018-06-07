# Usage

This folder contains files used to generate plots showing performance of oboe, with experiment design or random method for selecting a subset of models to start from. `Restricted' in filenames means only a subset of algorithms are selected from.

1. Oboe with experiment design
```
bash oboe_restricted.sh <path_to_folder_containing_preprocessed_datasets> <number_of_datasets_running_with_oboe_concurrently>
```
will generate `.pkl` files containing performance of oboe on a collection of datasets and output them into a separate folder, placed in the same folder as the dataset folder.

2. Oboe with random
Change line 6 of `oboe_restricted.py` to `automl_path = 'oboe_random/automl'`, and the rest are the same as above.
