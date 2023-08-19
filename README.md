# SLAN
Switch LSTM Aggregate Network (SLAN) to model Irregulary Sampled Time Series (ISTS)

# Datasets 

## Raw Data

MIMIC: We don't provide the original data for MIMIC-III in this repo as it violates the Data Usage Agreement. You can get access to MIMIC dataset following the instruction provided [here](https://github.com/kaggarwal/ClinicalNotesICU#steps) .

P12 and P19 : You can get access to raw P12/P19 data via https://physionet.org/content/challenge-2012/1.0.0/ and https://physionet.org/content/challenge-2019/1.0.0/ . Alternatively, you can use `medical_ts_datasets` from (SeFT github repository)[https://github.com/BorgwardtLab/Set_Functions_for_Time_Series/tree/master#available-datasets].


## Data Preprocessing

MIMIC : Follow the notebook `notebooks/getting_mimic_stats.ipynb` to generate the stat files.

P12 and P19:  Follow the notebooks `notebooks/getting_p12.ipynb` and `notebooks/getting_p12.ipynb` for preprocessing and generating stat files.


## Processed Data
Required
1. We don't have the permission to share the MIMIC data. So, you have to download the MIMIC data from the official link given in the Raw Data section. Do the following steps to get the MIMIC data setup for SLAN.
 - Download MIMIC data folder and store it in the `Data` Folder. Link to download MIMIC data folder https://figshare.com/s/4dacf4af493b79311fd7
 - Download the Raw MIMIC data from the official link. Store all the files in the test folder to the test folder of the above downloaded folder.
 - Store all the files in the train folder to the train folder of the above downloaded folder.
2. Download P12 data and store it in the `Data` Folder. Link to download P12 data - https://figshare.com/s/cb14d27aa3c86e97d4f1
3. Download P19 data and store it in the `Data` Folder. Link to download P19 data - https://figshare.com/s/a56fdf085943234a12e1

# Libraries required
1. Run the code - `pip install -r requirements.txt`

# To run the code
1. For MIMIC - `python code/main_MIMIC.py`
2. For P12 - `python code/main_P12.py`
3. For P19 - `python code/main_P19.py`
4. To Check saved result - `python code/read_results.py`

- There are various arguments that you can pass along with the above command.
Check code.md file for a detailed code arguments and discussion.