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

Terminology:

**IDPs** – Internally Displaced Person/People, contextually meaning anyone who has been forced to leave their home as a result to avoid armed conflict in this case but still reside within the Ukranian borders.

**Humanitarian Condition Level** - based on the European’s INFORM Severity index which determines a level 1-5, based on:
- Impact of crisis
- People in need
- Condition of people
  
**Access to humanitarian needs** - Healthcare, shelter, food, etc.

`# male population` and `# total older population (60 years and up)` provided a stark constrast in the data, since Ukraine restricted men from the ages 18-60 from leaving the borders in case of a need for fighters. 

The `# registered IDPs` feature was used as a label for the supervised time series model.

We chose this particular dataset because its dataset size was bigger than others that were provided through HDX, had features necessary for our approach, and was representative of all of the oblasts and their categorical features.


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

Since there were many features that correlated to each geographical situation within each oblast, we wanted to observe broader patterns within our dataset through clustering, and then further used them for a fixed supervised learning numerical prediction for IDPs. 

#### Unsupervised Learning : K-Means Clustering 

K-Means Clustering gathers data points that are similar to each other in some shape or form, and groups them into each clusters, and measures the data points based upon the sum of the squared distances between each point and the mean of its assigned cluster.

Visualized through PCA (Principal Component Analysis), we used the elbow method to find the optimal value of the hyperparameter `K`, as `K = numbers of clusters`. The elbow method finds where the rate of decrease sharply changes within the plotted cluster, minimizing the total variance within each cluster. The optimal number was 4 different clusters.

<img width="568" alt="image" src="https://github.com/Aaleia/Guidehouse-1D/assets/143746727/11a1e7fd-1859-4717-ac43-fc0e0f85bf7e">

<img width="403" alt="image" src="https://github.com/Aaleia/Guidehouse-1D/assets/143746727/cc9a7801-cc80-4b38-80e4-4fa4171eed5a">

#### Individual Cluster Analysis
Each cluster and the data points within them was expressed through the individual oblasts it was tied to.

<img width="913" alt="image" src="https://github.com/Aaleia/Guidehouse-1D/assets/143746727/57690e64-5baa-450c-a01a-fae007f94745">


**Cluster 0:** Cherkaska, Chernihivska, Chernivetska, Ivano-Frankivska, Khmelnytska, Kirovohradska, Mykolaivska, Poltavska, Rivnenska, Sumska, Ternopilska, Vinnytska, Volynska, Zakarpatska, Zhytomyrska

**Cluster 1:** Khersonska, Luhanska, Zaporizka

**Cluster 2:** Dnipropetrovska, Kharkivska, Kyiv, Kyivska, Lvivska, Odeska

**Cluster 3:** Donetska

**Donetska has its own cluster reserved for itself, because while cluster 2 contains Kyiv, the captial of Ukraine, Donetska holds the largest population with the steepest decrease in IDPs. According to geographical data driven from the dataset, Russia has focused most of its efforts within this region, due to its proximity to the border.** 

We utilized a heatmap correlation matrix for each cluster, in order to find a broader pattern between IDPs and the categorical features it related to. **Red** represents a very strong correlation, **Gray** is neutral, and **Blue** represents a very weak correlation. `# female older population (60 years and up)`, `# people affected`, `# km^2 unconfirmed control`, `# km^2 controlled by Ukrainian authorities/forces`, `# male population`, and `# total older population (60 years and up)` had the most correlation to `# registered IDPs`overall. `# km^2 controlled by Russian forces` was used as a control variable in order to compared the two controlled regions against each other when it came to displacement.


