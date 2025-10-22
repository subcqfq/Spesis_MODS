## PT-MLP·LSTM-eICU
This model utilizes pre-training with integrated MIMIC-IV and eICU datasets and employs a high- and low-frequency data separation strategy.

## MLP·LSTM
No pre-training is applied, but the high- and low-frequency data separation strategy is retained. The training data consists of 2,802 trauma sepsis patients from MIMIC-IV.

## PT-MLP-eICU
Incorporate pre-training but does not use high- and low-frequency separation. Both the 34 low-frequency features and the 7 high-frequency time-series features at the final moment of the time window were used as inputs to the MLP model. 

## PT-LSTM-eICU
Similar to the PT-MLP-eICU structure, this model replaces the MLP with an LSTM. The low-frequency data were interpolated to a 4-hour resolution, meaning that the value remained constant within each 4-hour window. These data, along with the high-frequency time-series data from the same 4-hour window, were then uniformly input into the LSTM model.

## PT-MLP·LSTM
Uses both high- and low-frequency separation strategy and pre-training, but the pre-training phase only leverages MIMIC-IV data (49,648 ICU patients) without incorporating eICU data.
