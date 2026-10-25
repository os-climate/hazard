The mean work loss indicator is calculated from the 'Wet Bulb Globe Temperature' (WBGT) indicator:

$$
I^\text{WBGT}_d = 0.567 \times T^\text{avg}_d + 0.393 \times p^\text{vapour}_d + 3.94
$$

$I^\text{WBGT}_d$ is the WBGT indicator, $T^\text{avg}_d$ is the average daily near-surface temperature (in degrees Celsius) on day index, $d$, and $p^\text{vapour}_d$
is the water vapour partial pressure (in kPa). $p^\text{vapour}$ is calculated from relative humidity $h^\text{relative}_d$ via:

$$
p^\text{vapour}_d = \frac{h^\text{relative}_d}{100} \times 6.105 \times \exp \left( \frac{17.27 \times T^\text{avg}_d}{237.7 + T^\text{avg}_d} \right)
$$

The work ability indicator, $I^{\text{WA}}_d$, is finally calculated via:

$$
I^{\text{WA}}_d = 0.1 + 0.9 / \left( 1 + (I^\text{WBGT}_d / \alpha_1)^{\alpha_2} \right)
$$

where $\alpha_1$ and $\alpha_2$ depend on the work intensity: $(32.98, 17.81)$ for low, $(30.94, 16.64)$ for medium and $(24.64, 22.72)$ for high intensity.

The mean annual work loss indicator, $i^{\text{WL}}$, is calculated via:

$$
I^{\text{WL}} = 1 - \frac{1}{n_y} \sum_{d = 1}^{n_y} I^{\text{WA}}_d, \qquad i^{\text{WL}} = \mathbb{E}(I^{\text{WL}}),
$$

$I^{\text{WL}}$ being the work loss in a single year and $n_y$ the number of days in the year. The expectation is estimated by averaging over a 20-year window. The OS-Climate-generated indicators are inferred from CMIP6 data, averaged over 6 models: ACCESS-CM2, CMCC-ESM2, CNRM-CM6-1, MPI-ESM1-2-LR, MIROC6 and NorESM2-MM.
The indicators are generated for periods: 'historical' (averaged over 1995-2014), 2030 (2021-2040), 2040 (2031-2050) and 2050 (2041-2060).
