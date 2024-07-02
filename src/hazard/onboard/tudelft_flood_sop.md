Maximum and minimum standards of protection for riverine floods occurring in Europe in the present and future climate. 
This is derived from a data set of protected flood extent i.e. the minimum return period for which flood depth is non-zero.

The data set is [here](https://data.4tu.nl/datasets/df7b63b0-1114-4515-a562-117ca165dc5b), part of the
[RAIN data set](https://data.4tu.nl/collections/1e84bf47-5838-40cb-b381-64d3497b3b36)
(Risk Analysis of Infrastructure Networks in Response to Extreme Weather). The RAIN report is
[here](http://rain-project.eu/wp-content/uploads/2016/09/D2.5_REPORT_final.pdf).

The 'River_flood_extent_X_Y_Z_with_protection' data set is obtained by applying FLOPROS database flood protection standards
on top of a data set of unprotected flood depth. That is, it provides information about flood protection for regions susceptible
to flood. This data set is not simply based on FLOPROS database therefore.

A minimum and maximum is provided to capture uncertainty in flood protection which vulnerability models may choose to take into
account. The protection bands are those of FLOPROS: 2–10 years, 10–30 years, 30–100 years, 100–300 years, 300–1000 years, 1000–10000 years.

As an example, if a region has a protected extent of 300 years, this is taken to imply that the area is protected against flood within the
bounds 100-300 years. It is then a modelling choice as to whether protection is taken from the 100 or 300 year flood depth.
