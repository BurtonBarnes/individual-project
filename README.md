# Individual Project
## Project Description

The second world war is known as the bloodiest conflict in human history. I have decided to look into the weather patterns of that time period, specifically the mean temperature. I will look into what features if any affect the MeanTemp of the dataframe.

# Project Goal

* Discover drivers for MeanTemp in the Summary_of_Weather.csv
* Use drivers to develop machine learning model to predict MeanTemp
* This information will be used to further our understanding of which elements contribute to or detract from a log_error.

# Initial Thoughts

* MeanTemp will be  driven my the month of the year
* Precipitation will be a driver of the MeanTemp
* Year will not affect the MeanTemp

# The Plan

* Aquire data

* Explore data in search of what causes MeanTemp
    * Answer the following initial questions
        * What is the distribution of MeanTemp?
        * Can the MeanTemp be predicted with the day?
        * Can the MeanTemp be predicted with the month?
        * Can the MeanTemp be predicted with the year?
        * Can the MeanTemp be predicted with the amount of precipitation?
        * Can the MeanTemp be predicted with the amount of snowfall?
        
* Develop a Model to predict MeanTemp
    * Use drivers identified in explore to build predictive models of different types
    * Evaluate models on train and validate data
    * Select the best model to use on test data
    
* Draw Conclusions

# Data Dictionary

| Feature | Definition |
|:--------|:-----------|
|STA| Weather Station|
|Date| The Date|
|Precip| Precipitation in mm|
|MaxTemp| Maximum temperature in degrees Celsius|
|MinTemp| Minimum temperature in degrees Celsius
|MeanTemp| Mean temperature in degrees Celsius
|Snowfall| Snowfall and ice pellets in mm|
|YR| Year of observation|
|MO| Month of observation|
|DA| Day of observation|

# Steps to Reproduce
1) Clone this repository
2) Ensure an .env file is in the repo
3) Put the data in  the file containing the cloned repo
4) Run notebook

# Takeaways and Conclusions
* The average temperature throughout the war was 8 degrees celsius
* You can predict the temperature with snowfall, the month, and the year
* Feature gathering of temperature comparing where it is taken

# Next Steps
* Comparing weather from then to now would yield siginicant results
* Features that include what type of environmnet the data was taken from such as arid, temperate or artic