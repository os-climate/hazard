Days per year for which the 'Wet Bulb Globe Temperature' indicator is above a threshold specified in °C:

$$I =  \frac{365}{n_y} \sum_{i = 1}^{n_y} \boldsymbol{\mathbb{1}}_{T^\text{WBGT}_i > T^\text{ref}}$$

The 'Wet-Bulb Globe Temperature' (WBGT) indicator is calculated from both the average daily near-surface surface temperature in in °C denoted $T^\text{avg}$ and the water vapour partial pressure in kPa denoted $p^\text{vapour}$:

$$
T^\text{WBGT}_i = 0.567 \times T^\text{avg}_i + 0.393 \times p^\text{vapour}_i + 3.94
$$

The water vapour partial pressure $p^\text{vapour}$ is calculated from relative humidity $h_r$:

$$
p^\text{vapour}_i = \frac{h_r}{100} \times 6.105 \times \exp \left( \frac{17.27 \times T^\text{avg}_i}{237.7 + T^\text{avg}_i} \right)
$$

The OS-Climate-generated indicators are inferred from CMIP6 data, averaged over 6 models: ACCESS-CM2, CMCC-ESM2, CNRM-CM6-1, MPI-ESM1-2-LR, MIROC6 and NorESM2-MM.
The indicators are generated for periods: 'historical' (averaged over 1995-2014), 2030 (2021-2040), 2040 (2031-2050) and 2050 (2041-2060).

The OS-Climate-generated indicators are inferred from downscaled CMIP6 data, averaged over for 6 Global Circulation Models: ACCESS-CM2, CMCC-ESM2, CNRM-CM6-1, MPI-ESM1-2-LR, MIROC6 and NorESM2-MM.
The downscaled data is sourced from the [NASA Earth Exchange Global Daily Downscaled Projections](https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6).
Indicators are generated for periods: 'historical' (averaged over 1995-2014), 2030 (2021-2040), 2040 (2031-2050) and 2050 (2041-2060).
