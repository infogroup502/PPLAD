# Privacy-Preserving Lightweight Time-Series Anomaly Detection for Resource-Limited Industrial IoT Edge Devices (TII 2025)

## Framework


## Requirements
The recommended requirements for PPLAD are specified as follows:
- torch==1.13.0
- numpy==1.26.4
- pandas==2.2.2
- scikit-learn==1.5.1
- matplotlib==3.9.2
- statsmodels==0.14.2
- tsfresh==0.20.3
- hurst==0.0.5
- arch==7.0.0

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```

## Data
The datasets can be obtained and put into datasets/ folder in the following way:
- Our model supports anomaly detection for multivariate time series datasets.
- We provide the SKAB dataset. If you want to use your own dataset, please place your datasetfiles in the `/dataset/<dataset>/` folder, following the format `<dataset>_train.npy`, `<dataset>_test.npy`, `<dataset>_test_label.npy`.

## Code Description
There are six files/folders in the source
- data_factory: The preprocessing folder/file. All datasets preprocessing codes are here.
- main.py: The main python file. You can adjustment all parameters in there.
- metrics: There is the evaluation metrics code folder.
- model: PPLAD model folder
- solver.py: Another python file. The training, validation, and testing processing are all in there
- requirements.txt: Python packages needed to run this repo

- ## Usage
1. Install Python 3.9, PyTorch >= 1.4.0
2. Download the datasets
3. To train and evaluate PPLAD on a dataset, run the following command:
```bash
python main.py 
```
