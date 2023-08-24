Days per year for which the average near-surface temperature 'tas' is above a threshold specified in °C.

$$
I =  \frac{365}{n_y} \sum_{i = 1}^{n_y} \boldsymbol{\mathbb{1}}_{\; \, T^{avg}_i > T^\text{ref}} \nonumber
$$
$I$ is the indicator, $T^\text{avg}_i$ is the daily average near-surface temperature for day index $i$ in °C, $n_y$ is the number of days in the year
and $T^\text{ref}$ is the reference temperature.
The OS-Climate-generated indicators are inferred from downscaled CMIP6 data. This is done for 6 Global Circulation Models: ACCESS-CM2, CMCC-ESM2, CNRM-CM6-1, MPI-ESM1-2-LR, MIROC6 and NorESM2-MM.
The downscaled data is sourced from the [NASA Earth Exchange Global Daily Downscaled Projections](https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6).
Indicators are generated for periods: 'historical' (averaged over 1995-2014), 2030 (2021-2040), 2040 (2031-2050)
and 2050 (2041-2060).