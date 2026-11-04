Days per year for which the daily-mean or daily-maximum 'Wet-Bulb Globe Temperature' (WBGT) is above a threshold specified in °C, and the equivalent for the working day (see below):

$$
I^\text{mean}(T^\text{ref}) = \frac{365}{n_y} \sum_{d = 1}^{n_y} \boldsymbol{\mathbb{1}}_{\; \, T^\text{WBGT,mean}_d > T^\text{ref}}, \qquad I^\text{max}(T^\text{ref}) = \frac{365}{n_y} \sum_{d = 1}^{n_y} \boldsymbol{\mathbb{1}}_{\; \, T^\text{WBGT,max}_d > T^\text{ref}}
$$

$d$ is the day index, $n_y$ is the number of days in the year and $T^\text{ref}$ is the threshold (reference) temperature. The indicators are the expectations of $I^\text{mean}$, $I^\text{max}$ and $I^\text{4+4+4}$ (below), estimated by averaging over a 20-year window. For each indicator, the index dimension of the array is the threshold.

The (in-shade) WBGT is calculated from the daily mean and daily maximum near-surface temperature in °C, $T^\text{avg}_d$ and $T^\text{max}_d$, and the water vapour partial pressure in hPa, $p^\text{vapour}_d$:

$$
T^\text{WBGT,mean}_d = 0.567 \times T^\text{avg}_d + 0.393 \times p^\text{vapour}_d + 3.94, \qquad T^\text{WBGT,max}_d = 0.567 \times T^\text{max}_d + 0.393 \times p^\text{vapour}_d + 3.94
$$

The water vapour partial pressure is calculated from the daily mean temperature and relative humidity $h^\text{relative}_d$ (in %) and is taken to be constant through the day, so that the same value is used for the daily maximum:

$$
p^\text{vapour}_d = \frac{h^\text{relative}_d}{100} \times 6.105 \times \exp \left( \frac{17.27 \times T^\text{avg}_d}{237.7 + T^\text{avg}_d} \right)
$$

The daily-mean and daily-maximum WBGT can be used to model hourly WBGT during the working day. The '4+4+4' method of the International Labour Organization (ILO, 'Working on a warmer planet', 2019, Appendix I) takes 12 daylight working hours, of which 4 are close to the maximum WBGT, 4 close to the mean WBGT and 4 halfway between the two, $T^\text{WBGT,half}_d = (T^\text{WBGT,mean}_d + T^\text{WBGT,max}_d) / 2$. The working-day indicator is the number of days per year for which WBGT is above the threshold, each day counting as the fraction of its working hours above the threshold:

$$
I^\text{4+4+4}(T^\text{ref}) = \frac{365}{n_y} \sum_{d = 1}^{n_y} \frac{1}{3} \left( \boldsymbol{\mathbb{1}}_{\; \, T^\text{WBGT,mean}_d > T^\text{ref}} + \boldsymbol{\mathbb{1}}_{\; \, T^\text{WBGT,half}_d > T^\text{ref}} + \boldsymbol{\mathbb{1}}_{\; \, T^\text{WBGT,max}_d > T^\text{ref}} \right)
$$

The working hours per year with WBGT above the threshold are $12 \, I^\text{4+4+4}$. Differences between successive thresholds give the hours within WBGT bands, to which exposure-response functions for work capacity (such as those of the ILO for work intensities of 200 W, 300 W and 400 W) can be applied. For other models of $m$ hours at the maximum and $n$ hours at the mean, the hours per year above a threshold are $m \, I^\text{max} + n \, I^\text{mean}$.
