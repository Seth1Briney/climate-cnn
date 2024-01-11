# climate-cnn

Predicts day of year given global temperature data. Utilizes aggressive summary statistics pooling  to reduce dimensionality.

The globe is split into 16 regions in latitute, and 32 regions in  longitute. For each region, statistical summary pooling is performed as a data preprocessing step, replacing that region with the min, max, median, mean, and std of the temperatures. This carries a significant dimensionality reduction. Data are saved in this form to reduce the overhead of loading the large data and performing statistical pooling repeatedly.
