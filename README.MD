## Overview
This projects impelments a Kalman filter-based model to aggregate election polling data. Including the weighing of pollsters based on their past performance.  


## How and why
It features a 1D random walk Kalman Filter for Hungarian poll results. Combined with smoothing to create trend visualizations that hopefully better capture the underlying dynamics, then just relying on one or two pollsters. I was inspired to use a Kalman Filter by Politico, who uses it for their Poll of Polls results.

As a source of data I rely on Vox Populi. 

## Results
![Kalman graph](Graphs/kalman_nyers.png)
![Kalman LOESS graph](Graphs/kalman_loess.png)



