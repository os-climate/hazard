cwlVersion: v1.2
$graph:
  - class: Workflow
    id: produce-hazard-indicator
    label: produce hazard indicator
    doc: Produce a hazard indicator with os_climate_hazard
    requirements:
      ResourceRequirement:
        coresMax: 2
        ramMax: 4096

    inputs:
      ceda_ftp_username:
        type: string
        default: ""
      ceda_ftp_password:
        type: string
        default: ""
      ceda_ftp_url:
        type: string
        default: ""
      source_dataset:
        type: string
      gcm_list:
        type: string
      scenario_list:
        type: string
      central_year_list:
        type: string
      central_year_historical:
        type: int
      window_years:
        type: int
      indicator:
        type: string
        default: "days_tas_above_indicator"
      store:
        type: string
        default: "./indicator"
      inventory_format:
        type: string
        default: "osc"

    outputs:
      - id: indicator-result
        type: Directory
        outputSource:
          - indicator-step/indicator-results

    steps:
      indicator-step:
        run: "#indicator-command"
        in:
          ceda_ftp_username: ceda_ftp_username
          ceda_ftp_password: ceda_ftp_password
          ceda_ftp_url: ceda_ftp_url
          source_dataset: source_dataset
          gcm_list: gcm_list
          scenario_list: scenario_list
          central_year_list: central_year_list
          central_year_historical: central_year_historical
          window_years: window_years
          indicator: indicator
          store: store
          inventory_format: inventory_format
        out:
          - indicator-results


  - class: CommandLineTool
    id: indicator-command

    hints:
      DockerRequirement:
        dockerPull: public.ecr.aws/c9k5s3u3/os-hazard-indicator:ukcp18compat

    requirements:
      ResourceRequirement:
        coresMax: 2
        ramMax: 4096
      NetworkAccess:
        networkAccess: true
      EnvVarRequirement:
          envDef:
            CEDA_FTP_USERNAME: $(inputs.ceda_ftp_username)
            CEDA_FTP_URL: $(inputs.ceda_ftp_url)
            CEDA_FTP_PASSWORD: $(inputs.ceda_ftp_password)

    inputs:
      ceda_ftp_username:
        type: string
      ceda_ftp_password:
        type: string
      ceda_ftp_url:
        type: string
      source_dataset:
        type: string
      gcm_list:
        type: string
      scenario_list:
        type: string
      central_year_list:
        type: string
      central_year_historical:
        type: int
      window_years:
        type: int
      indicator:
        type: string
      store:
        type: string
      inventory_format:
        type: string

    outputs:
      indicator-results:
        type: Directory
        outputBinding:
          glob: "./indicator"

    baseCommand: os_climate_hazard

    arguments:
      - valueFrom: $(inputs.indicator)
      - prefix: --store
        valueFrom: $(inputs.store)
      - prefix: --source_dataset
        valueFrom: $(inputs.source_dataset)
      - prefix: --gcm_list
        valueFrom: $(inputs.gcm_list)
      - prefix: --scenario_list
        valueFrom: $(inputs.scenario_list)
      - prefix: --central_year_list
        valueFrom: $(inputs.central_year_list)
      - prefix: --central_year_historical
        valueFrom: $(inputs.central_year_historical)
      - prefix: --window_years
        valueFrom: $(inputs.window_years)
      - prefix: --inventory_format
        valueFrom: $(inputs.inventory_format)
