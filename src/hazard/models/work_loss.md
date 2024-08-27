The mean work loss indicator is calculated from the 'Wet Bulb Globe Temperature' (WBGT) indicator:

$$
I^\text{WBGT}_i = 0.567 \times T^\text{avg}_i + 0.393 \times p^\text{vapour}_i + 3.94
$$

$I^\text{WBGT}_i$ is the WBGT indicator, $T^\text{avg}_i$ is the average daily near-surface surface temperature (in degress Celsius) on day index, $i$, and $p^\text{vapour}$
is the water vapour partial pressure (in kPa). $p^\text{vapour}$ is calculated from relative humidity $h_r$ via:

$$
p^\text{vapour}_i = \frac{h_r}{100} \times 6.105 \times \exp \left( \frac{17.27 \times T^\text{avg}_i}{237.7 + T^\text{avg}_i} \right)
$$

The work ability indicator, $I^{\text{WA}}$ is finally calculated via:

$$
I^{\text{WA}}_i = 0.1 + 0.9 / \left( 1 + (I^\text{WBGT}_i / \alpha_1)^{\alpha_2} \right)
$$

An annual average work loss indicator, $I^{\text{WL}}$ is calculated via:

$$
I^{\text{WL}} = 1 - \frac{1}{n_y} \sum_{i = 1}^{n_y} I^{\text{WA}}_i,
$$

$n_y$ being the number of days in the year.
