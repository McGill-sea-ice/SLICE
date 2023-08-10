# St-Lawrence Ice Condition Evolution (SLICE)

## Overview
This project aims at forcasting the freeze-up date (FUD) in the St-Lawrence River near Montreal (Longueuil) at sub-seasonal time scales (weeks to months). Two approaches are developped: a multiple linear regression model that directly predicts the FUD using monthly average or accumulated predictors, and an LSTM Encoder-Decoder (LSTM-ED) model that indirectly predicts the FUD from water temperature forecasts.

Both models require predictor data from ERA5 and SEAS5. The LSTM-ED model additionally requires daily water temperature data from the Longueuil water filtration plant and a daily time series of the NAO index.

The observed FUD is defined from the Longueuil water temperature as the first day when Tw < 0.75 degC. The time series of FUD detected from Tw can be compared against visual observations of FUD recorded by the St-Lawrence Seaway Management Corporation (SLSMC) and ice charts from the Canadian Coast Guard (CCG). 

## Installation 
Start by cloning the repository:

```bash
# Check if your SSH key is configured on GitHub
ssh -T git@github.com
# Clone the project
git clone git@github.com:McGill-sea-ice/slice.git 
```

This project uses a [**conda environment**][conda]. Start by accessing the project folder:

[conda]: https://docs.conda.io/en/latest/miniconda.html

```bash
cd slice 
```

Create and activate the project's environment (this installs the dependencies):

```bash
conda env create -f environment.yml
conda activate MLgpu
```

## Folder organization
The repository contains three main folders: 


- `cameras/`: Codes for downloading and saving hourly videos from the CCG cameras looking at the St-Lawrence River.


- `data_prep/`: Codes for fetching, pre-processing, and saving predictor data.


- `analysis/`: Codes for fetching, pre-processing, and saving predictor data.

Helper functions are saved in `functions.py`, `functions_ML.py`, `functions_MLR.py`, and `functions_encoderdecoder.py`. 

**Note**: If a folder `other/` exists somewhere, it usually contains codes that were used for preliminary analysis or analyses that are not used anymore. These folders and the routines they contain and are not needed in the pipeline for generating the models and the predictions.

## Data
The routines in `data_prep/` and `analysis/` are used to prepare the data, build the model, and get FUD predictions. To run them, you need to copy the `data/` folder that is located on `crunch` at `/storage/amelie/slice/data/`.

## Usage
Below is a step-by-step description of what routines to run in order to get the data, process it, build the models, and get predictions.









## i) Data Processing
### 1. Get raw data
**Note**: we always use region 'D'.

- ##### 1.1 Observed FUD (from SLSMC and charts)
    No code. Obtained as csv file from SLSMC, or CCG ice charts.


- ##### 1.2. ERA5 (hourly)
    data_prep/get-era5-data.py`: Downloads raw nc data from CDS store. 
    

- ##### 1.3. SEAS5 (daily) 
    `data_prep/get-seas5-data.py` & `data_prep/get-seas51-data.py`: Downloads raw nc data from CDS store. 

    **Note**: `data_prep/get-seas51-data.py` is to be used to download SEAS5.1 for 2022 and later.


- ##### 1.4 Tw (daily)
    No code. Obtained as a csv file from Longueuil water filtration plant. This data will become automatically saved on `crunch` when the new water temperature measuring station is installed. 


- ##### 1.5 Water levels & discharge (daily)
    Downloaded from Internet at: https://wateroffice.ec.gc.ca/mainmenu/historical_data_index_e.html
    


- ##### 1.6 Climate indices: 
    Downloaded from Internet. 

   -  Daily NAO, AO, PNA: https://ftp.cpc.ncep.noaa.gov/cwlinks/
    - Monthly EA, SCAND, PolEur, WP, EP-NP, PNA, TNH, PT: https://www.cpc.ncep.noaa.gov/data/teledoc/telecontents.shtml
    - Monthly AMO: http://www.psl.noaa.gov/data/timeseries/AMO/
    - Monthly SOI: https://psl.noaa.gov/data/climateindices/list/
    - Monthly ONI: https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt
    - Monthly PDO (HadISST 1.1): https://psl.noaa.gov/pdo/ 
    


### 2. Raw data processing
- ##### 2.1 Observed FUD (from SLSMC and charts) 
    - `data_prep/freezeup_dataset_preparation.py` takes raw csv data of merged (from SLSMC and CCG ice charts) dates of first ice, dates of stable ice, and freeze-up dates observed from charts, and saves it to numpy format. Saves the data `data/processed/freezeup_dates_SLSMC/freezeup_SLSMC_[LOCATION].npz`

    - `data_prep/freezeup_finer_dataset_preparation.py` uses the csv files of the refined analysis of ice charts with dates of first available char,date of last day with open water from charts, and dates of first ice on chart for multiple locations near the Seaway and saves it to numpy format. Saves data `data/processed/freezeup_dates_charts/freezeup_charts_[LOCATION].npz`

- ##### 2.2 ERA5
    `data_prep/ERA5_data_preparation_with_dailysum.py` reads raw ERA5 nc files and makes daily average or daily sum time series from the raw hourly data using CDO. Saves `ERA5_[DAILYSUM OR DAILYMEAN]_[VARNAME].npz`


- ##### 2.3 SEAS5 
    `data_prep/SEAS5_data_preparation_with_dailyincrement.py` reads raw SEAS5 nc files and makes time series of daily values of snowfall, runoff, and total_precipitation, instead of time series of accumulation from beginning of forecast. Saves the file with the same name and `_processed_` appended to the feature so we know it has been modified from its raw values.


- ##### 2.4 Tw 
    - First, run `data_prep/Twater_cities_data_preparation.py` to take raw csv data and save it to numpy format. Saves the data `data/processed/Twater_cities/Twater_cities_[LOCATION].npz`.

    - Then run `data_prep/Twater_cleaning_filters.py` to filter outliers (e.g. steps and lines etc.) and fill data gaps hosrter than 7 days. Saves the data `data/processed/Twater_cities/Twater_cities_[LOCATION]_cleaned_filled.npz` and `data/processed/Twater_cities/Twater_cities_all_[LOCATIONS]_updated_cleaned_filled.npz`.

    - To check if there is an offset in the water temperature time series to take into consideration, run `data_prep/Twater_winter_offset_analysis.py`. This is verified using the winter months when temperature is supposed to be at zero degC. If it the mean winter water temperature is larger than 0 +/- 0.5 degC, then we use an offset to bring back the time series to realistic values. If an offset is found to be needed, then `Twater_cleaning_filters.py` should be re-runned to save the water temperature time series with the according offset. This script does not save data.

    - After data has been cleaned and filled and offset, use `data_prep/Twater_premerging_analysis.py` if you want to check that the distribution of Tw from two different cities or stations are statistically equivalent. This script does not save data.


- ##### 2.5 Water levels & discharge
    `data_prep/water_level_data_preparation.py` takes raw csv data of water level and discharge and transfers it to daily time series in numpy format. Saves `water_levels_discharge_[LOCATION].npz`


- ##### 2.6 Climate indices
    - Run `data_prep/ONI_monthly_prep.py` and `data_prep/PDO_monthly_prep.py` to convert raw csv data of monthly ONI and PDO indices to daily time series in numpy format. Saves `ONI_index_monthly.npz` and `PDO_index_monthly_hadisst1.npz`

    - Run `data_prep/multi_indices_daily_prep.py`, `data_prep/multi_indices_monthly_prep.py`, and `data_prep/multi_indices_12monthsformat_monthly_prep.py` to convert raw csv data of various monthly/daily indices (NAO, AO, PNA, AMO, SOI, WP,TNH, EA, POLEUR, PT, SCAND, EPNP, EA) to daily time series in numpy format. These scripts save the files `[INDEX]_index_monthly.npz` & `[INDEX]_index_daily.npz`




### 3. Save predictor data table with all processed predictors (for usage with LSTM-ED model)

- ##### 3.1 
    Run `data_prep/make_ML_dataset.py` to load daily time series of Tw, ERA5 predictors, discharge, water levels, and climate indices, and saves a new npz with all variables in the same numpy array. Saves `ML_dataset.npz` that is used in `save_predictor_daily_timeseries.py` below.
   

- ##### 3.2
    Run `data_prep/save_predictor_daily_timeseries.py` to load Tw time series, detect FUD, create new categorical variables (true if FUD happens wihtin the next XX days), load predictor data from `ML_dataset_with_cansips.npz` and save everything in one DataFrame. Saves `predictor_data_daily_timeseries.xlsx` that is used in `encoderdecoder_lstm.py`
    
    **Note**: A commented draft for computing the time series of the equivalent internal energy of the ice-water system and ice temperature is also found in `data_prep/save_predictor_daily_timeseries.py`.   

    **Note**: The original data `ML_dataset_with_cansips.npz` that is needed in `data_prep/save_predictor_daily_timeseries.py` was originally generated by the script `analysis/MLR/stepwise_MLR/MLR_predictor_preselection_with_cansips.py`. The new script `data_prep/make_ML_dataset.py` (see above) is now doing the same thing, but without cansips variables because they are no longer needed.







## ii) Models and analysis
**Note**: We always use `freezeup_opt = 1` 

### 1. Multiple Linear Regression forecasts

- ##### 1.1 Save monthly predictors
    MLR models use monthly predictors. These have to be saved first. Run `analysis/MLR/all_combinations_monthly_predictor/save_monthly_predictors.py` to save monthly predictors from daily time series. Saves `data/monthly_predictors/monthly_vars_[LOCATION].npz` (**Note** this script and this step could go with data processing instead... )

- ##### 1.2 Check significant predictor correlation
    Use `analysis/MLR/all_combinations_monthly_predictor/monthly_pred_corr_FUD.py` to check month-by-month correlation coefficient of monthly predictors with FUD. This script generates the correlation coefficient table in the paper. Does not save data.

- ##### 1.3 MLR prefect forecast experiment and model selction
    Run `MLR_perfect_forecast_model_selection_LOOkCV.py` to build and evaluate MLR models in leave-one-out with k-fold cross-validation (LOOk) for all possible combinations of predictors that showed significant correlation with the FUD (as per above step). Saves `analysis/MLR/all_combinations_monthly_predcitors/output/all_coefficients_significant_05/MLR_monthly_pred_varslast6monthsp05only_Jan1st_maxpred4_valid_scheme_LOOk.npz`

- ##### 1.4 MLR real-world forecast experiment & Categorical Baseline Forecast
    Finally, run `MLR_realworld_forecast_all_members.py` to run the chosen MLR model using the SEAS5 seasonal forecast predictors in a real-world forecast experiment. This script also evalutes the probabilistic forecasts using the MCRPSS and spread-error coefficient. The categorical baseline forecast accuracy using SEAS5 seasonal air forecasts for December is also computed here.


### 2. LSTM-ED forecasts

- ##### 2.1 LSTM-ED forecasts
    Run `encoderdecoder_lstm.py` to run the LSTM-ED model in standard or LOOk cross-validation (CV). Saves the loss and the model output Tw predictions for the train, valid, and test sets in `analysis/ML/encoderdecoder_src/output/` (one file per year when using LOOk, or one file only when using standard CV). The training checkpoints (`.ckpt.index` and `.ckpt.data` files; one per epoch) and the trained models (`.keras`; one per year if using LOOk or one per run if using standard CV) are saved in `analysis/ML/encoderdecoder_src/models/`. 

- ##### 2.2 Plot FUD and Tw evaluation metrics (CV and test sets)
    Run `compare_encoderdecoder_output.py` to visualize the Tw or FUD predictions and their evaluation metrics on the validation or test sets, for one or multiple LSTM-ED runs. Saves the FUD predictions and metrics in `analysis/ML/encoderdecoder_src/metrics/` (one file per run). 

- ##### 2.3 Plot FUD metrics (test set only) to compare with MLR and baseline forecasts 
    Run `plot_eval_metrics.py` to compare the deterministic FUD metrics of LSTM-ED runs against climatology and MLR results. This output the figures that are used in the paper (time series of FUD prediction in perfect forecast and real-world deterministic forecast; and 6-panel figure of metrics comparing ML, MLR, and climatology).
 
