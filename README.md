# climate-cnn

Predicts day of year given global temperature data. Utilizes aggressive summary statistics pooling  to reduce dimensionality.

The globe is split into 16 regions in latitute, and 32 regions in longitute (from 128x256 input). For each region, statistical summary pooling is performed as a data preprocessing step, replacing that region with the min, max, median, mean, and std of the temperatures. This carries a significant dimensionality reduction. Data are saved in this form to reduce the overhead of loading the larger data and performing statistical pooling repeatedly.

Targets (days of the year) are given circular encodings, so that 0 and 364 are very close.

The best model I trained with this was able to predict the day of the day of the year within 2 days.
