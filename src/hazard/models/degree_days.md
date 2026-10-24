Degree days indicators are calculated by integrating over time the amount by which the temperature
of the medium exceeds a reference temperature. The exact method of calculation may vary;
here the daily maximum near-surface temperature 'tasmax' is used to calculate an annual indicator:

$$
I^\text{dd} = \frac{365}{n_y} \sum_{d = 1}^{n_y} \max \left( T^\text{max}_d - T^\text{ref}, 0 \right), \qquad i^\text{dd} = \mathbb{E}(I^\text{dd}) \nonumber
$$

$i^\text{dd}$ is the indicator and $I^\text{dd}$ the degree days in a single year, $T^\text{max}_d$ is the daily maximum near-surface temperature for day index $d$, $n_y$ is the number of days in the year
and $T^\text{ref}$ is the reference temperature given in the indicator name (32°C by default).

The expectation is estimated by averaging over a 20-year window.
