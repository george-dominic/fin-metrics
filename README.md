# Fin Metrics
 
Machine learning model to understand the relationship between Stock Percent Change and Financial Key Metrics over time and predict stock price movement based on growth of the financial metrics

## Table of Contents
- Architecture
- Tech stack
- Packages used
- Pre-requisites
- Installation
- Usage
- Project Report
- Contributing
- Contact


## [Architecture](https://www.figma.com/file/eszLgfciBKqcBLn9SAwqUI/Pipeline?node-id=0%3A1&t=MeOWrWVQwHW2X8Rb-1)
![image](https://user-images.githubusercontent.com/94735949/233217585-7f0beb2b-522d-411d-85e4-81af8c504cfa.png)

## Tech Stack
- Python
- Mage
- OpenBB
- GCP BigQuery
- H2O AutoML
- Git

## Packages used
- pandas
- mage-ai
- google-cloud-bigquery
- openbb
- numpy
- lightgbm
- scikit-learn
- xgboost
- h2o


## Pre-requisites
- **Python**: Make sure you have Python installed on your machine. You can download the latest version of Python from the official Python website: python.org

- **Java**: Make sure you have Java installed on your machine. You can get the latest version from Java website: java.com

- **Package Manager**: Install a package manager like pip or conda. Pip is the default package manager for Python and is usually pre-installed with Python. Conda is an alternative package manager that is often used for managing Python environments. You can install conda by downloading Anaconda or Miniconda from their respective websites.

- **Virtual Environment (optional)**: It is recommended to set up a virtual environment for your Python project. Virtual environments allow you to isolate project dependencies and avoid conflicts between different projects. You can create a virtual environment using tools like venv, virtualenv, or conda.

- **Data Pipelines**: Some knowledge of Data Engineering and Data Pipelining concepts. On top of that, familarity with Mage Data would be beneficial


## Installation
1. Clone this repository 
```
git clone https://github.com/george-dominic/fin-metrics.git
```
2. Change into the project's directory
```
cd fin-metrics
```
3. Install the required dependencies by running the following command:
```
pip install -r requirements.txt
```

## Usage
1. Once you are inside fin-metrics directory, initialise the mage pipeline using:
```
mage start metrics
```
This will open up Mage UI on your local browser : https://localhost:6789

2. Click on pipelines on the left tab to view all the pipelines

3. Inside pipelines you can view actions performed within each block of the pipeline

## Project Report
You can peruse the complete project report [here](https://georgedominic.com/Fin-metrics-ML-666e04246cc4450d84816bbf668f60f9)

## Contributing
Efforts to better this project are most welcome, please follow these steps:

1. Fork this repository and clone it to your local machine.
2. Create a new branch for your feature or bug fix
```
git checkout -b your-branch-name
```
3. Make your changes and commit them with descriptive commit messages.
4. Push your changes to your forked repository:
```
git push origin your-branch-name
```
5. Open a pull request in this repository, describing your changes in detail.

## Contact
If you have any questions, suggestions or feedback, feel free to [hit me up](https://georgedominic.com/hmu) 

---

Thanks you! 😄
