# Forecast

## Installation

Please clone the repository via SSH or HTTPs.

### Docker

For launching experiments, please

1. Install Docker following the documentation https://docs.docker.com/get-docker/.
1. For Linux, please execute the post-installation
   https://docs.docker.com/engine/install/linux-postinstall/.
1. Enter the root of the repository `forecast-bcl/`.
1. Login to Docker Repo. You need to provide your git username/password (or AccessToken) for
   authentification.

   ```bash
   make login
   ```

1. Build docker image

   ```bash
   make setup-local  # with GPU
   make setup-no-gpu # without GPU
   ```

1. Enter the docker container

   ```bash
   make bash
   ```

1. Install pyrosetta if needed
   ```bash
   conda install pyrosetta -c https://levinthal:paradox@conda.graylab.jhu.edu -y
   ```

### Code Development

1. Install the conda environment
   ```bash
   conda env create -f environment.yml
   ```
1. Activate the created conda environment:
   ```bash
   conda activate forecast_bcl
   ```
1. Install pre-commit hooks
   ```bash
   pre-commit install
   ```

### Data Analysis

#### Pyrosetta Interface Analysis

```bash
make setup-local
make bash
analysis_pyrosetta --input_dir xx --output_dir xx --num_workers 4
```

After Performing the pyrosetta analysis, results can be joined into single file using the analysis
utils:

```bash
analysis_utils --pyrosetta --peptide_names xx --s3_input_path xx --s3_output_path xx
```

#### Arpeggio Interaction Analysis

```bash
make setup setup-arpeggio
make bash-arpeggio
bash gnn/analysis/arpeggio/install.sh
analysis_arpeggio --input_dir xx --output_dir xx --num_workers 4
```

### Training GCN models

#### Preprocessing

Start by preprocessing the dataset by running:

```bash
process --metadata_path pdb_data --data_name pdb_data --pdb_path pdb_data --train_path HLA-A02:01.train_train.csv --val_path HLA-A02:01.train_val.csv --test_path HLA-A02:01.train_test.csv --graph_type peptide
```

For aggregating data :

```bash
aggregate --metadata_path pdb_data --data_name pdb_data --pdb_path pdb_data --train_path HLA-A02:01.train_train.csv --val_path HLA-A02:01.train_val.csv --test_path HLA-A02:01.train_test.csv --graph_type peptide
```

\_\_Note: Check with the team members the path of the train_path, train_path and train_path
[here](DATA.md). You can find also already processed data.

#### Training

Train your GCN model by running:

```bash
train --metadata_path cache/pdb_tar --data_name pdb_tar --graph_type peptide --model GCN  --csv_label_train cache/pdb_tar/df_label_train.csv --csv_label_val cache/pdb_tar/df_label_val.csv --poses "relax" --epochs 100 --patience 10
```

#### Inference

Test your pre-trained model by running:

```bash
score --metadata_path cache/pdb_tar --data_name pdb_tar --graph_type peptide --model GCN --csv_label cache/pdb_tar/df_label_test.csv --poses "relax" --checkpoint_path out/path_to/checkpoint.pt
```
