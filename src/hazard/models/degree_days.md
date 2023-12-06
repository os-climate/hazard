Degree days indicators are calculated by integrating over time the absolute difference in temperature
of the medium over a reference temperature. The exact method of calculation may vary;
here the daily maximum near-surface temperature 'tasmax' is used to calculate an annual indicator:

$$
I^\text{dd} = \frac{365}{n_y} \sum_{i = 1}^{n_y} |  T^\text{max}_i - T^\text{ref} | \nonumber
$$

$I^\text{dd}$ is the indicator, $T^\text{max}$ is the daily maximum near-surface temperature, $n_y$ is the number of days in the year and $i$ is the day index.
and $T^\text{ref}$ is the reference temperature of 32°C. The OS-Climate-generated indicators are inferred
from downscaled CMIP6 data, averaged over 6 models: ACCESS-CM2, CMCC-ESM2, CNRM-CM6-1, MPI-ESM1-2-LR, MIROC6 and NorESM2-MM.
The downscaled data is sourced from the [NASA Earth Exchange Global Daily Downscaled Projections](https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6).
The indicators are generated for periods: 'historical' (averaged over 1995-2014), 2030 (2021-2040), 2040 (2031-2050)
and 2050 (2041-2060).
