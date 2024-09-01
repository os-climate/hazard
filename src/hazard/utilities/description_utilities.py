from typing import Iterable


def get_indicator_period_descriptions(
    scenarios: Iterable[str],
    central_year_historical: int,
    central_years: Iterable[int],
    window_years: int,
) -> str:
    base_description = "Indicators are generated for periods: "
    periods = []
    if "historical" in scenarios:
        periods.append(
            f"'historical' {_get_period_years(year=central_year_historical, historical=True, window_years=window_years)}"
        )
    for year in central_years:
        periods.append(
            f"{year} {_get_period_years(year=year, historical=False, window_years=window_years)}"
        )
    if len(periods) == 1:
        period_descriptions = base_description + periods[0] + "."
    elif len(periods) == 2:
        period_descriptions = base_description + periods[0] + " and " + periods[1] + "."
    else:
        period_descriptions = base_description
        for idx, period in enumerate(periods):
            last = idx == len(periods) - 1
            period_descriptions += (
                f"{'and ' if last else ''}{period}{'.' if last else ', '}"
            )
    return period_descriptions


def _get_period_years(year: int, historical: bool, window_years: int) -> str:
    window_years_max = year + window_years // 2 + (window_years % 2)
    window_years_min = year - window_years // 2
    return f"({'averaged over ' if historical else ''}{window_years_min}-{window_years_max})"
