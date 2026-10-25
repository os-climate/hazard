Degree days indicators are calculated by integrating over time the amount by which the temperature
of the medium exceeds a reference temperature. The exact method of calculation may vary;
here the daily average near-surface temperature 'tas' is used to calculate an annual indicator for each of
a set of threshold (reference) temperatures, 16°C, 20°C and 24°C by default:

$$
I^\text{dd} = \frac{365}{n_y} \sum_{d = 1}^{n_y} \max \left( T^\text{avg}_d - T^\text{ref}, 0 \right), \qquad i^\text{dd} = \mathbb{E}(I^\text{dd}) \nonumber
$$

$i^\text{dd}$ is the indicator and $I^\text{dd}$ the degree days in a single year, $T^\text{avg}_d$ is the daily average near-surface temperature for day index $d$, $n_y$ is the number of days in the year
and $T^\text{ref}$ is the threshold temperature.

The expectation is estimated by averaging over a 20-year window.
