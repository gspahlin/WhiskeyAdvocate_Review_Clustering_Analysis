# Whiskey_Analysis
Unsupervised learning and analysis with my whiskey database

The goal of this analysis is to understand what kinds of whiskeys are available in the market - in order to provide recommendations. My underlying hypothesis is that 
whiskeys will cluster by price, flavor characters, and abv in ways that are not captured by categories such as 'single malt scotch' or 'bourbon/tenessee' which are 
largely geographical in nature. I am using the t-SNE algorythm in Scikit-Learn to visualize the structure of the data in my feature space. The next step is to use a 
clustering algorythm to predict clusters in the data. I will use my best t-SNE parameters to visualize the clusters, and basic data analysis to learn the characteristics
of the clusters and find representatives. 
