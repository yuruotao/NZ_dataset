# Power Transportation Synthesized Dataset

## Introduction

Here contains the raw data and scripts for data processing and visualization for paper "[Power Transportation Synthesized Dataset]()". The data is deposited at [Figshare](), here we include the analysis of example data (South Island of New Zealand) for demonstration. The tree structure of this repository is shown below.

```yaml
    Power_Transportation_Synthesized_Dataset
    ├── data				# Contain the data to be analyzed (download from figshare)
    │    ├── behavior			# Behavior data from Scientific Data
    │    ├── boundaries			# Boundary shapefiles
    │    ├── charging_stations		# Charging stations in New Zealand
    │    ├── example			# Contain the data of south island of New Zealand
    │    ├── service_stations		# Service stations in New Zealand
    │    ├── state_highway		# New Zealand state highway shapefile
    │    └── ... 
    ├── ninja_accounts
    │    └── ninja_accounts.xlsx        # Excel for solar data crawler
    ├── result                     	# Directory to contain the results
    │    └── ...
    ├── scripts  
    │    ├── behavior.py		# Analyze the behavior of EV drivers
    │    ├── charging_stations.py	# Project the charging stations to highway and service stations
    │    ├── flow.py			# Imputation for flow data
    │    ├── registration.py		# New Zealand EV registration data analysis
    │    ├── service.py			# Estimate the charging demand for service stations
    │    ├── solar.py			# Obtain the solar potential for each service station
    │    └── optimization.py		# Optimization model for NZ infrastructures
    ├── main.py				# Analysis flow of the dataset
    ├── LICENSE
    ├── requirements.txt		# Dependencies for the project
    └── README.md
```

The sources of data are summarized here.

|         Data         |                                                            Data Source                                                            |
| :------------------: | :-------------------------------------------------------------------------------------------------------------------------------: |
|    City Districts    |                                                               [NZTA]()                                                               |
|      Coastlines      |                     [LINZ](https://data.linz.govt.nz/layer/51560-nz-coastlines-and-islands-polygons-topo-1500k/)                     |
|  Charging Stations  |                       [NZTA](https://opendata-nzta.opendata.arcgis.com/maps/238dd4298c0445d8ac8567eefe22413e)                       |
|  Highway Structure  |                             [LINZ](https://data.linz.govt.nz/layer/50329-nz-road-centrelines-topo-150k/)                             |
|   Service Stations   |                                          [Z Energy](https://www.z.co.nz/find-a-station/#/)                                          |
|   Solar Potential   |                                         [Renewables Ninja](https://www.renewables.ninja/#/)                                         |
|     Vehicle Flow     |                                                           [Waka Kotahi](https://opendata-nzta.opendata.arcgis.com/datasets/41e05dcdfcb749d390f7785543fb3b14/about)                                                           |
| Vehicle Registration | [Ministry of Transport](https://www.transport.govt.nz/statistics-and-insights/fleet-statistics/sheet/low-emissions-vehicle-report#/) |
|  Charging Behavior  |                               [Scientific Data](https://www.nature.com/articles/s41597-024-02942-9#/)                               |
|  Electricity Price  |                                                [EMI](https://www.emi.ea.govt.nz/#/.)                                                |

## Usage Note

To run the code, you need to first download the code and data from figshare, move the figshare data into folder "data", install the dependencies in "requirements.txt", then run script "main.py". If you need to use the web crawler, you firstly need to fill in the Excel "ninja_accounts.xlsx".

This repository is under MIT License, please feel free to use. If you find this repository helpful, please cite the following bibtex entry:

```
@article{,
  title={},
  author={yuruotao},
  year={2024}
}
```

## Contact

For questions or comments, you can reach me at [yuruotao@outlook.com](yuruotao@outlook.com).
