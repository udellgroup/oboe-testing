# Usage

This folder contains necessary files to pre-process datasets for further experimental evaluation. By default, these scripts first one-hot-encode categorical features and then standardize all features.

1. OpenML datasets
First, do
```
python select_datasets.py
```
with the dataset selection criteria specified in line 15 of  `select_datasets.py`. This operation selects OpenML datasets based on the specified criteria, outputs corresponding indices into a csv file and selection criteria into a txt file; their filenames are the same and are specified in line 17 of `select_datasets.py`.

Then, do
```
./generate_preprocessed_dataset.sh
```
This generates preprocessed OpenML datasets, whose indices come from `selected_OpenML_classification_dataset_indices.csv`, by default. Output directory should is specified in line 12 of  `generate_preprocessed_dataset.py` and is set to be `selected_OpenML_classification_datasets` by default.

Another useful operation is
```
python process_csv.py <filename>.csv
```
This outputs dataset indices contained in `<filename>.csv` with space segments (instead of line breaks), making it easy to write bash script for iterating over datasets.

2. UCI datasets
Thanks to <https://github.com/JackDunnNZ/uci-data>, we are able to convert UCI datasets to a format easier to pre-process.

Do
```
python uci-dataset-preprocessing.py
```
This preprocesses datasets in `.data` format outputted by the execution of code in the above GitHub repository, and store them in folder `uci_datasets`.

