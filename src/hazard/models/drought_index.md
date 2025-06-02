Expected months per year for which the 12 month Standardized Precipitation Evapotranspiration Index (SPEI) is below a threshold.

$$
i = \mathbb{E} \left( \sum_{i = 1}^{12} \mathbb{1}_{\; \, S_i < s^\text{thresh}} \right) \nonumber
$$

$i$ is the indicator, $S_i$ is the 12 month SPEI for the month with index $i$.
and $s^\text{thresh}$ is the threshold index.

The expectation is estimated by averaging over a 20-year window.
