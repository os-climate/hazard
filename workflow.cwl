cwlVersion: v1.2

$namespaces:
  s: https://schema.org/
s:softwareVersion: 0.1
schemas:
- http://schema.org/version/9.0/schemaorg-current-http.rdf
$graph:
  - class: Workflow

    id: produce-indicator
    label: produce hazard indicator
    doc: produce hazard indicator

    requirements:
      ResourceRequirement:
        coresMax: 2
        ramMax: 4096
    
    inputs: 
      gcm_list:
        type: string
        default: "[NorESM2-MM]"

    outputs:
      - id: indicator-result
        type: Directory
        outputSource:
          - indicator-step/indicator-results

    steps:
      indicator-step:
        run: "#indicator-command"
        in:
          gcm_list: gcm_list
        out:
          - indicator-results

  - class: CommandLineTool
    id: indicator-command

    requirements:
      ResourceRequirement:
        coresMax: 2
        ramMax: 4096
      NetworkAccess:
        networkAccess: true

    hints:
      DockerRequirement:
        dockerPull: public.ecr.aws/c9k5s3u3/os-hazard-indicator

    baseCommand: ["os_climate_hazard", "days_tas_above_indicator", "--inventory_format", "stac", "--store", "./indicator", "--"]
   
    arguments: []
    
    inputs:
      gcm_list:
        type: string
        inputBinding:
          prefix: --gcm_list
          separate: true
    
    outputs:
      indicator-results:
        type: Directory
        outputBinding:
          glob: "./indicator"