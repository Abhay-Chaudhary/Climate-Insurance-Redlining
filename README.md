# Who’s Being Priced Out of Protection? Predicting FAIR Plan Enrollment in California

### Group Members
Leonardo Barleta, Abhay Chaudhary, Allison Lucas, and Tiana Townsend

### Introduction
California's residential insurance market faces unprecedented disruption as major insurers retreat due to escalating wildfire risk, construction cost increases, and volatile reinsurance markets. State Farm, the state's largest residential insurer, suspended new policy issuance in 2023 and did not renew about 72,000 existing policies. Insurers are now passing higher premiums to consumers and pushing homeowners toward the California FAIR Plan—a state-backed insurance program that provides basic property coverage when traditional insurers refuse to write policies. In Sacramento County, FAIR Plan usage more than doubled in one year. With FAIR Plan resources limited (~$377 M liquidity, as of Jan 2025), regulators and advocates warn of an “uninsurable future” for wildfire-exposed homeowners.
This analysis examines whether FAIR Plan enrollment patterns and financial liability can be predicted using disaster risk exposure, socioeconomic indicators, and insurance cost metrics. Developing a reliable forecasting model would enable policymakers, community organization, and homeowners to anticipate coverage gaps, allocate resources proactively, and identify communities vulnerable to insurance market disruption.
### Data Collection
We compiled a comprehensive dataset combining FAIR Plan data for 2022 with publicly available information about residential insurance policies, housing prices, socioeconomic conditions, and climate-related disasters. This data was retrieved from the following sources (a longer description of the data sources is available on the data_prep.ipynb notebook):

* American Community Survey - reported income, race, and housing conditions
* Zillow Housing Value Index
* Governor-proclaimed disasters from 1991 to present
* California Department of Insurance residential policy data, based on insurance company bi-annual reports
    
The primary geographic unit of analysis is the ZIP Code Tabulation Area (ZCTA), which aligns with the resolution of most available datasets. We predict the usage of FAIR Plan policies in a given ZIP Code, measured in percentage of residential units. As a secondary target, we also estimate the financial impact of the program’s expansion, based on total exposure covered in FAIR Plan policies. This variable measures the maximum potential loss California could face in case of climate disaster.
We focus on the 2018–2021 period, where data from multiple sources are available and comparable. Given that detailed FAIR Plan data is only available for 2022, we computed the rate of growth of a given indicator (e.g., growth in house prices) for the four previous years for some key variables.
### Modeling Approach
EDA and Feature Engineering: We performed common EDA tasks, such as determining p-values and feature importance via Random Forest modeling. We engineered features from the cleaned dataset to transform data into more informative and suitable features. Two examples of these features are Renewal Resilience, the proportion of policies successfully renewed, and Housing Value to Median Income Ratio, both created by combining other features. The final collection of all features was then sorted by p-value or feature importance and redundancies were removed with lower p-values and higher feature importance taking precedence. Additionally, visual elements such as a correlation matrix and plots of the main features overlaid on a map of California were helpful in preparing our most indicative features for model selection.
Model Selection: we developed different features sets based on exploratory data analysis and use cases that we envision that our project could provide insight, as well as compare predicting techniques to produce the optimal regressor. We compared linear regression, kNN regression, Gradient Boosting, XGBoost, and logistic regression (in this case, using the prediction probabilities) to a baseline model (mean) using cross-validation. Our metrics were MSE, RMSE, MAE and, for exposure, the total monetary amount underwritten in FAIR Plan policies.

The most performant model significantly improved baseline predictions for FAIR Plan usage, as measured by MSE. It uses Gradient Boosting on the “main_features” set, which includes important variables detected in EDA. However, models only using the top 2 predictors ('Renewal Resilience' and 'Change in Earned Premiums') performed very similarly. The sets “dynamic_features” and “non_ins_%_fair”–respectively, features that capture changes over time and features not using data reported by insurance companies–had the worst performance. A visual inspection of our predictions, as presented in the `04_analysis.ipynb`, suggests that our models have been able to capture the spatial patterns in the data.

The secondary model predicting total exposure also improved baseline estimates, nearly halving the difference between the actual values underwritten in FAIR Plan policies and their predictions. In absolute values, our predictions were $19 billion dollars (compared to $35 billions of baseline estimates) off, from a total of $164 bi in exposure covered in California-backed insurance policies. However, this specific model is not very stable and would require additional tuning for a more realistic assessment.


## Software Requirements
The project was built using Python 3.11+. Required packages include:

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `xgboost`
* `geopandas`
* `pathlib`
* `scipy.stats`

## Deliverables
- `Who's Being Priced Out of Protection - Executive Summary.pdf`: Detailed overview of findings, methods, and policy implications.
- `03_model_selection.ipynb`, `04_analysis.ipynb`: All analysis and modeling steps in reproducible Jupyter Notebooks.
- `Who's Being Priced Out of Protection - Slides.pdf`: Summary slides for project presentation.
