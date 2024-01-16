# Predicting Ukraine's Emerging Humanitarian Needs, Guidehouse-1D
## Fall 2023, AI Studio Project Write-Up

### Forecasting Ukraine displacement, utilizing K-Means Clustering and Seasonal ARIMAX :

### Business Focus 

As a direct result of 2022 Russia's invasion in Ukraine, the rate of displacement has been at of an all time high, especially in vulnerable oblasts within km^2 range of the border. This project focuses on the consulting firm Guidehouse's cliente request to implement forecasting machine learning models in order to predict present humantarian needs in Ukraine. The goal was to cluster patterns within the ACAPS Master Dataset within each oblast's displacement and other features, and to then utilize time series for an estimated future displacement fixed number. Results are only attained to the geographical conditions located in the ACAPS Master Dataset and should be used as a supplement tool for humantarian action.

### Data Preparation and Validation
#### DATASET DESCRIPTION

We explored HDX and used the ACAPS Master Dataset, which spans monthly from **Jan 2022 - Sep 2023**. We explored HDX and used the ACAPS Master Dataset, which spans monthly from **Jan 2022 - Sep 2023**. The dataset focuses on Ukranian civilians from 24 different Oblasts + Kyiv (special administrative status), that correlated to each categorical citizen and geographical related features, such as `# male younger population (0 - 14 years)` and `# total older population (60 years and up)` providing a stark constrast in the data, since Ukraine restricted men from the ages 18-60 from leaving the borders in case of a need for fighters. The `# registered IDPs` feature was used as a label for the supervised time series model.

