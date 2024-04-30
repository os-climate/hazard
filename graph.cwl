cwlVersion: v1.2

$namespaces:
  s: https://schema.org/
s:softwareVersion: 1.2
schemas:
- http://schema.org/version/9.0/schemaorg-current-http.rdf
$graph:
  - class: Workflow

    id: produce-indicator
    label: produce hazard indicator
    doc: produce hazard indicator

    requirements:
      ResourceRequirement:
        coresMax: 4
        ramMax: 8192
    
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
        coresMax: 4
        ramMax: 8192
      NetworkAccess:
        networkAccess: true

    hints:
      DockerRequirement:
        dockerPull: local-hazard-image-test

    baseCommand: ["os_climate_hazard", "days_tas_above_indicator", "--store", "./indicator", "--"]
   
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