# Predicting Ukraine's Emerging Humanitarian Needs, Guidehouse-1D
## Fall 2023, AI Studio Project Write-Up

### Forecasting Ukraine displacement, utilizing K-Means Clustering and Seasonal ARIMAX :

### Business Focus 

As a direct result of 2022 Russia's invasion in Ukraine, the rate of displacement has been at of an all time high, especially in vulnerable oblasts within km^2 range of the border. This project focuses on the consulting firm *Guidehouse's* clientele request to implement forecasting machine learning models in order to predict present humantarian needs in Ukraine. The goal was to cluster patterns within the *ACAPS Master Dataset* within each oblast's displacement and other features, and to then utilize time series for an estimated future displacement fixed number. Results are only attained to the geographical conditions located in the *ACAPS Master Dataset* and should be used as a supplement tool for humantarian action.

### Data Preparation and Validation
#### DATASET DESCRIPTION

We explored HDX and used the *ACAPS Master Dataset*, which spans monthly from **Jan 2022 - Sep 2023**. We explored HDX and used the ACAPS Master Dataset, which spans monthly from **Jan 2022 - Sep 2023**. The dataset focuses on Ukranian civilians from 24 different Oblasts + Kyiv (special administrative status), that correlated to each categorical citizen and geographical related features, such as `# km^2 controlled by Russian forces`, `# civilian fatalities`, etc. 

Primarly we focused on these features for feature selection:

- Area controlled by Ukraine and Russia​
- Population & people exposed​
- People affected​
- Demographics​
- Internally Displaced People (IDPs)​
- Levels of severity (1-5)​
- Minimal, stressed, moderate, severe, extreme
- Violence and fatalities​
- Unemployment​

`# male population` and `# total older population (60 years and up)` provided a stark constrast in the data, since Ukraine restricted men from the ages 18-60 from leaving the borders in case of a need for fighters. 

The `# registered IDPs` feature was used as a label for the supervised time series model.

#### *ACAPS MASTER DATASET* PREPROCESSING

Before preprocessing we made the following adjustments :

- Removed all non-numerical values (e.g. postal code), except for Oblast (region)​
- Removed data on wages, income, pension, and inflation​ since it was not relevant
- Did not use food/fuel cost data, as it was too likely to be affected by other factors​
- Removed data not available for every Oblast​
- Removed data from totality countries features (Crimea and Ukraine)​

And we used the following methods : 

**Time Scaling** - Implemented a Time column, based on days since Jan 2022 using the monthly data, in preparation for the time series model.

**Standardization** - Standard Scaler for all numerical features through Z-score normalization (all eligible columns except Oblast)
Standard scaler: mean value 0 and standard deviation 1 (removing the mean and and scaling to unit variance) for better PCA centroid performance to handle variance within the same scale.

**Dummy/Indicator Variables** - Converting each Oblast column to numerical by giving each oblast a number for time series.

### Choosing the Model

Since there were many features that correlated to each geographical situation within each oblast, we wanted to observe patterns within our dataset through clustering, and then further used them for a fixed supervised learning numerical prediction for IDPs. 

#### Unsupervised Learning : K-Means Clustering 
