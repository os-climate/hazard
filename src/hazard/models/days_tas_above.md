Days per year for which the daily-mean near-surface temperature 'tas' or the daily-maximum near-surface temperature 'tasmax' is above a threshold specified in °C:

$$
I^\text{mean}(T^\text{ref}) = \frac{365}{n_y} \sum_{d = 1}^{n_y} \boldsymbol{\mathbb{1}}_{\; \, T^\text{avg}_d > T^\text{ref}}, \qquad I^\text{max}(T^\text{ref}) = \frac{365}{n_y} \sum_{d = 1}^{n_y} \boldsymbol{\mathbb{1}}_{\; \, T^\text{max}_d > T^\text{ref}}
$$

$T^\text{avg}_d$ and $T^\text{max}_d$ are the daily mean and daily maximum near-surface temperature in °C for day index $d$, $n_y$ is the number of days in the year and $T^\text{ref}$ is the threshold (reference) temperature. The indicators are the expectations of $I^\text{mean}$ and $I^\text{max}$, estimated by averaging over a 20-year window. For each indicator, the index dimension of the array is the threshold.
