FROM --platform=linux/amd64 python:3.10

WORKDIR /app

COPY pyproject.toml pyproject.toml

COPY src src

RUN python -m pip install -e .

CMD ["os_climate_hazard"]