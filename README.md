# SLAN
Switch LSTM Aggregate Network (SLAN) to model Irregulary Sampled Time Series (ISTS)

# Datasets 

## Raw Data

## Data Preprocessing

## Processed Data
Required
1. We don't have the permission to share the MIMIC data. So, you have to download the MIMIC data from the official link given in the Raw Data section. 
We have shared a folder where you need to store some 
Download MIMIC data and store it in the `Data` Folder. Link to download MIMIC data - https://figshare.com/s/22b79cc75d7374642d0a
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