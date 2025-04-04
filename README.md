# Power Transportation Synthesized Dataset

## Introduction

Here contains the raw data and scripts for data processing and visualization for paper "[High-resolution multi-source traffic data in New Zealand](https://www.nature.com/articles/s41597-024-04067-5)". The data is deposited at [Figshare](https://springernature.figshare.com/articles/dataset/High-resolution_multi-source_traffic_data_a_case_study_in_New_Zealand/26965246), here we include the analysis of example data (South Island of New Zealand) for demonstration. The tree structure of this repository is shown below.

```yaml
    High-resolution multi-source traffic data in New Zealand
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
|  City Districts  |                                                             [NZTA](https://datafinder.stats.govt.nz/)                                                             |
|    Coastlines    |                   [LINZ](https://data.linz.govt.nz/layer/51560-nz-coastlines-and-islands-polygons-topo-1500k/)                   |
| Highway Structure |                           [LINZ](https://data.linz.govt.nz/layer/50329-nz-road-centrelines-topo-150k/)                           |
|   Vehicle Flow   |             [Waka Kotahi](https://opendata-nzta.opendata.arcgis.com/datasets/41e05dcdfcb749d390f7785543fb3b14/about)             |
|      Weather      |                             [NOAA](https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/)                             |
|      Holiday      | [MBIE](https://www.employment.govt.nz/leave-and-holidays/public-holidays/previous-years-public-holidays-and-anniversary-dates#/) |
|  Extreme Weather  |                                                 [NIWA](https://hwe.niwa.co.nz/ )                                                 |

The data downloaded from Figshare should be grouped as below. For shapefiles, the .shp document is the one you interact with. For the usage of other documents, you can visit [this page](https://en.wikipedia.org/wiki/Shapefile) 
```yaml
    Data downloaded from Figshare
    ├── data                            # Contain the data to be analyzed (download from figshare)
         ├── city_districts             # Boundary shapefiles
         │   ├── city_districts.cpg     
         │   ├── city_districts.dbf     
         │   ├── city_districts.prj     
         │   ├── city_districts.shp     # Document to interact with
         │   ├── city_districts.xml 
         │   ├── city_districts.txt     # Specifications of data    
         │   ├── city_districts.sbx     
         │   ├── city_districts.pdf     # Comments for shapefile
         │   ├── city_districts.shx
         │   └── city_districts.sbn
         ├── state_highway              # New Zealand state highway shapefile
         │   ├── state_highway.cpg      
         │   ├── state_highway.dbf      
         │   ├── state_highway.prj      
         │   ├── state_highway.shp      # Document to interact with
         │   ├── state_highway.shp.xml  
         │   └── state_highway.cpg      
         └── NZDB/NZDB.db               # SQLite3 database, can be viewed with vscode plugins or softwares like sqlitebrowser
                                        # The tables and keys are documented in paper
```

## Usage Note

To run the code, you need to first download the code and data from figshare, move the figshare data into folder "data", install the dependencies in "requirements.txt", then run script "main.py" for distribution visualization, run script "flow_process.py" for the process of flow data, and script "weather_process.py" for the process of weather data. Note that some of the code may require modification since this script analyzes the raw data instead of the data provided in database format. If you intend to replicate the analysis, you can run the "weather_data_obtain.py" to obtain the weather data, and download all the data from their source summarized above.

This repository is under MIT License, please feel free to use. If you find this repository helpful, please cite the following bibtex entry:

```
@article{li2024high,
  title={High-resolution multi-source traffic data in New Zealand},
  author={Li, Bo and Yu, Ruotao and Chen, Zijun and Ding, Yingzhe and Yang, Mingxia and Li, Jinghua and Wang, Jianxiao and Zhong, Haiwang},
  journal={Scientific Data},
  volume={11},
  number={1},
  pages={1216},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## Contact

For questions or comments, you can reach me at [yuruotao@outlook.com](yuruotao@outlook.com).
