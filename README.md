# Probabilistic Artificial Intelligence 2023 - ETH

## Task 1: Gaussian Process Regression

The task was to predict the pollution at different locations in a 2-dimensional map. Additionally, there are areas where we cannot underpredict the pollution.
This is modelled within the cost function, where a wrong prediction normally simply is calculated as L2-loss, however if we underpredict in an important area, we incur a way larger cost (weighted 50 instead of 1).

To solve this task, we mainly tried a lot of different kernels for the GPRegressor (Matern, RGB, Linear and combinations of those).
To make sure we never underpredict we additionally add an additional factor of the standard deviation (in our case a factor of 1.75 led to the best result) to important locations.
To additionally improve our position on the leaderboard, we then used multiple different GP regressors for the different locations on the grid.
In our case we simply split the area into 4 equally sized regions and predict each on its own.

The picture below visualizes the predictions. One can clearly see the regions where we are not allowed to underpredict (the circles which appear slightly lighter).
![](./task1/extended_evaluation.png "Visualization of the predictions")

## Task 2: SWA-Gaussian

In this task, we had to predict aerial images of landscapes into 6 different classes. Additionally there are training samples which do not belong to a certain class and are classified as "ambiguous".
The loss of the model is then calculated as a combination of the ECE (Expected Calibration Error) and a loss based on the predictions, where predicting ambiguous (e.g., -1) incurs a fixed cost and wrongly predicting a non-ambiguous sample incurs a larger fixed cost.

To solve this we implemented SWAG according to the paper linked in the project description ([A Simple Baseline for Bayesian Uncertainty in Deep Learning](https://arxiv.org/pdf/1902.02476.pdf)).
We first implemented SWAG diagonal, to get a first working solution, which was sufficient to pass the medium baseline, and afterwards we implemented SWAG-Full to pass the hard baseline as well.
Fixing the prediction threshold to $\frac{2}{3}$ led to the best results on the leaderboard.

The pictures below show the most and least confident predictions and additionally a reliability diagram for our model.
![](./task1/examples_most_confident.png "Visualization of the most confident predictions")
![](./task1/examples_least_confident.png "Visualization of the least confident predictions")
![](./task1/reliability_diagram.png "Reliability diagram")

## Task 3: Bayesian Optimization

## Task 4: Reinforcement Learning
