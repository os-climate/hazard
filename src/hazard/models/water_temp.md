Weeks per year for which the average water temperature is above a threshold specified in °C:

$$
I = \frac{52}{n_y} \sum_{w = 1}^{n_y} \boldsymbol{\mathbb{1}}_{\; \, T^\text{avg}_w > T^\text{ref}}, \qquad i = \mathbb{E}(I) \nonumber
$$

$i$ is the indicator and $I$ the number of weeks above the threshold in a single year, $T^\text{avg}_w$ is the weekly average water temperature for week index $w$ in °C, $n_y$ is the number of weeks in the year
and $T^\text{ref}$ is the reference temperature. The expectation is estimated by averaging over the years of each period listed below.

The OS-Climate-generated indicators are inferred from downscaled CMIP5 data. This is done for 5 Global Climate Models: GFDL-ESM2M, HadGEM2-ES, IPSL-CM5A-LR, MIROC-ESM-CHEM and NorESM1-M.
The downscaled data is sourced from the [Futurestreams dataset](https://geo.public.data.uu.nl/vault-futurestreams/research-futurestreams%5B1633685642%5D/original/waterTemp/) on the data publication platform of Utrecht University.
Indicators are generated for periods: 'historical' (averaged over 1976-2005), 2020 (2006-2030), 2030 (2021-2040), 2040 (2031-2050), 2050 (2041-2060), 2060 (2051-2070), 2070 (2061-2080), 2080 (2071-2090) and 2090 (2081-2100).
