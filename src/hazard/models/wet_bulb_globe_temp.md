Days per year for which the 'Wet Bulb Globe Temperature' indicator is above a threshold specified in °C:

$$
I =  \frac{365}{n_y} \sum_{i = 1}^{n_y} \boldsymbol{\mathbb{1}}_{\; \, T^\text{WBGT}_i > T^\text{ref}} \nonumber
$$

$I$ is the indicator, $n_y$ is the number of days in the sample and $T^\text{ref}$ is the reference temperature.

The 'Wet-Bulb Globe Temperature' (WBGT) indicator is calculated from both the average daily near-surface surface temperature in °C denoted $T^\text{avg}$ and the water vapour partial pressure in kPa denoted $p^\text{vapour}$:

$$
T^\text{WBGT}_i = 0.567 \times T^\text{avg}_i + 0.393 \times p^\text{vapour}_i + 3.94
$$

The water vapour partial pressure $p^\text{vapour}$ is calculated from relative humidity $h^\text{relative}$:

$$
p^\text{vapour}_i = \frac{h^\text{relative}_i}{100} \times 6.105 \times \exp \left( \frac{17.27 \times T^\text{avg}_i}{237.7 + T^\text{avg}_i} \right)
$$
