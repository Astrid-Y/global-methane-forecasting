# Methane concentrations forecasting based on ConvLSTM with attention

## Table of Contents

1. [ Introduction ](#introduction)
2. [ Datasets](#datasets)
3. [ Usage ](#usage)
4. [ Documentation ](#documentation)
5. [ License ](#license)
6. [ References ](#references)


## Introduction

This project proposes a model that combines Convolutional Long Short-Term Memory (ConvLSTM) with attention mechanisms to solve the challenge of traditional physical models that require large computation and high time cost, and improve the accuracy of methane concentration predictions.


## Datasets

### Download data after preprocessing

You can download the preprocessed dataset in [Google drive share folder](https://drive.google.com/drive/folders/1fLCjwfHpkhVkHlmgPF6dtGZ4APgMBIf4?usp=sharing).

The dataset contains three files and should be included in the same directory:

```
rootdir
  ├── emissions.nc
  ├── meteorology.nc
  └── total_column_methane.nc
```

You are not suggested to change the names of the files, otherwise you need to modify the path to load data in [src/ch4forecast/utils/dataloader.py](https://github.com/ese-msc-2023/irp-bh223/blob/main/src/ch4forecast/utils/dataloader.py)

### Original data

If you want to explore the original dataset, you can download data from the official websites, the information of datasets are as follows:

| Data Type |  Details | Description | Link |
|----------|-----------|----------|----------|
| Emissions | CH4 anthropogenic | 0.1 × 0.1, monthly | [CAMS-GLOB-ANT v2.1](https://eccad.sedoo.fr/#/metadata/479) |
| Emissions | CH4 wetland fluxes | 0.5 × 0.5, monthly | [LPJ-HYMN climatology (1990-2008)](https://earth.gov/ghgcenter/data-catalog/lpjeosim-wetlandch4-grid-v2) |
| Emissions | CH4 Fire | 0.1 × 0.1, daily | [GFAS v1.2](https://eccad.sedoo.fr/#/metadata/404) |
| Meteorology | 10m u-component of wind<br>10m v-component of wind<br>2m temperature<br>Surface Geopotential<br>Total column water vapour   | 0.75° x 0.75°, 3-hourly | [CAMS global reanalysis (EAC4)](https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-global-reanalysis-eac4?tab=overview) |
| Concentration | Total column methane | 0.75° x 0.75°, 3-hourly | same above |

This repository provides some notebooks to deal with the original datasets in folder `dataprocess/`:
1. [resample/resampling.ipynb](https://github.com/ese-msc-2023/irp-bh223/blob/main/dataprocess/resample/resampling.ipynb): To resample the all the data  into the same resolution and sampling frequence.
2. [visualize/explore.ipynb](https://github.com/ese-msc-2023/irp-bh223/blob/main/dataprocess/visualize/explore.ipynb): To draw the snapshot of the dataset in a specific timepoint, or to draw an animation of the change of data in a range of time.

## Usage

### Installation

To set up this project, follow these steps:

1. Clone the repository

    ```bash
    git clone https://github.com/ese-msc-2023/irp-bh223.git
    ```

2. Create a conda environment

    ```bash
    conda create -n env_name python=3.10
    conda activate env_name
    ```

3. Install the ch4forecast package
   
    ```bash
    cd /path/to/project
    pip install -e .
    ```

### Train and test the model

#### Train the model

Modify the parameters of `data_path` and `output_path`, and then run the command:

```bash
ch4forecast --train --n_epoch 30 --model CSLSTM --name CS --data_path /directory/to/data --output_path /directory/to/output
```

You can also modify other parameters, see [src/ch4forecast/main.py](https://github.com/ese-msc-2023/irp-bh223/blob/main/src/ch4forecast/main.py) in details.

#### Evaluate the model performance

Modify the parameters of `data_path`, `output_path` and `pretrained`, and then run the command:

```bash
ch4forecast --test --test_days 14 --model CSLSTM --name CS --pretrained /path/to/model_weights.pth --data_path /directory/to/data --output_path /directory/to/output
```

You can download a pretrained model weights: [CS_30](https://drive.google.com/file/d/1Ec9pC1GK-IIeAS2zBHdp8rZtGARebW7o/view?usp=sharing), which is trained 30 epochs with model CSLSTM.

Notes: When testing, the parameters that control the model should be the same as that of the pretrained model.

## Documentation

To access the documentation that have generated using Sphinx, please go to the [docs/build/html/index.html](https://github.com/ese-msc-2023/irp-bh223/blob/main/docs/build/html/index.html) file.

Alternatively, on mac, please run the following command:

`open docs/build/html/index.html`


## License

This project is licensed under the MIT Licence. See the [LICENCE](https://github.com/ese-msc-2023/irp-bh223/blob/main/LICENSE) file for details.


## References

All references including AI tools that have been used in coding are in the [Reference.md](https://github.com/ese-msc-2023/irp-bh223/blob/main/Reference.md) file.