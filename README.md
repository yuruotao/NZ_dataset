# Power Transportation Synthesized Dataset

## Introduction

Here contains the raw data and scripts for data processing and visualization for paper "[High-resolution synthetic highway traffic data: a case study in New Zealand]()". The data is deposited at [Figshare](), here we include the analysis of example data (South Island of New Zealand) for demonstration. The tree structure of this repository is shown below.

```yaml
    High-resolution synthetic highway traffic data: a case study in New Zealand
    ├── data				# Contain the data to be analyzed (download from figshare)
    │    ├── boundary			# Boundary shapefiles
    │    ├── state_highway		# New Zealand state highway shapefile
    │    └── NZDB/NZDB.db		# Database
    ├── result                     	# Directory to contain the results
    │    └── ...
    ├── utils  
    │    ├── basic_statistics.py	# Script for calculating the basic statistics
    │    ├── database_upload.py		# Create the database NZDB.db from multiple data sources
    │    ├── flow_process.py		# Script for flow data processing
    │    ├── visualization.py		# Visualization functions
    │    ├── weather_data_obtain.py	# Script for weather data obtain
    │    └── weather_process.py		# Script for weather data processing
    ├── main.py				# Visualize the distribution for flow and weather stations
    ├── LICENSE
    ├── requirements.txt		# Dependencies for the project
    └── README.md
```

The sources of data are summarized here.

|       Data       |                                                          Data Source                                                          |
| :---------------: | :---------------------------------------------------------------------------------------------------------------------------: |
|  City Districts  |                                                             [NZTA]()                                                             |
|    Coastlines    |                   [LINZ](https://data.linz.govt.nz/layer/51560-nz-coastlines-and-islands-polygons-topo-1500k/)                   |
| Highway Structure |                           [LINZ](https://data.linz.govt.nz/layer/50329-nz-road-centrelines-topo-150k/)                           |
|   Vehicle Flow   |             [Waka Kotahi](https://opendata-nzta.opendata.arcgis.com/datasets/41e05dcdfcb749d390f7785543fb3b14/about)             |
|      Weather      |                             [NOAA](https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/)                             |
|      Holiday      | [MBIE](https://www.employment.govt.nz/leave-and-holidays/public-holidays/previous-years-public-holidays-and-anniversary-dates#/) |
|  Extreme Weather  |                                                 [NIWA](https://hwe.niwa.co.nz/ )                                                 |

## Usage Note

To run the code, you need to first download the code and data from figshare, move the figshare data into folder "data", install the dependencies in "requirements.txt", then run script "main.py" for distribution visualization, run script "flow_process.py" for the process of flow data, and script "weather_process.py" for the process of weather data. Note that some of the code may require modification since this script analyzes the raw data instead of the data provided in database format. If you intend to replicate the analysis, you can run the "weather_data_obtain.py" to obtain the weather data, and download all the data from their source summarized above.

This repository is under MIT License, please feel free to use. If you find this repository helpful, please cite the following bibtex entry:

```
@article{,
  title={High-resolution synthetic highway traffic data: a case study in New Zealand},
  author={yuruotao},
  year={2024}
}
```

## Contact

For questions or comments, you can reach me at [yuruotao@outlook.com](yuruotao@outlook.com).
