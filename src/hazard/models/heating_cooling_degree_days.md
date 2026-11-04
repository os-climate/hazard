Degree days indicators are calculated by integrating over time the amount by which the temperature is above (cooling degree days) or below (heating degree days) a reference temperature. Here the daily mean near-surface temperature 'tas' is used to calculate annual indicators for each of a set of reference temperatures:

$$
I^\text{cdd}(T^\text{ref}) = \frac{365}{n_y} \sum_{d = 1}^{n_y} \max \left( T^\text{avg}_d - T^\text{ref}, 0 \right), \qquad I^\text{hdd}(T^\text{ref}) = \frac{365}{n_y} \sum_{d = 1}^{n_y} \max \left( T^\text{ref} - T^\text{avg}_d, 0 \right)
$$

$T^\text{avg}_d$ is the daily mean near-surface temperature in °C for day index $d$, $n_y$ is the number of days in the year and $T^\text{ref}$ is the reference temperature. The indicators are the expectations of $I^\text{cdd}$ and $I^\text{hdd}$, estimated by averaging over a 20-year window. For each indicator, the index dimension of the array is the reference temperature. Heating degree days are given for reference temperatures from 10 °C, representative of well-insulated buildings with high internal gains, to 22 °C; cooling degree days for 18 °C to 24 °C.
