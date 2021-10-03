# Whiskey_Analysis
Unsupervised learning and analysis with my whiskey database
<br><br>
The goal of this analysis was to understand what kinds of whiskeys are available in the market - in order to provide recommendations. My underlying hypothesis is that 
whiskeys will cluster by price, flavor characters, and abv in ways that are not captured by categories such as 'single malt scotch' or 'bourbon/tenessee' which are 
largely geographical in nature. I am using the t-SNE algorythm in Scikit-Learn to visualize the structure of the data in my feature space. The feature spaces I used in
this analysis all come from the review language present on WhiskeyAdvocate.com. They were constructed using word counts of discriptive words in the reviews. After 
visualization of the data, the next step is to use a clustering algorythm to predict clusters in the data. I will use my best t-SNE parameters to visualize the clusters, 
and basic data analysis to learn the characteristics of the clusters and find representatives. 
<br><br>
Step 1 - Optimizing the visualization
<br>
In this study I was interested in using unsupervised clustering methods to identify similar whiskeys. I wanted some kind of visualization technique, however, in order to 
see what my clustering algorythm was actually doing. I chose t-SNE for this, which is a dimensionality reduction technique that relies on a distance metric to determine
where points should fall in relation to one another in a 2 dimensional space. The first code I commited to this repository was in the service of optimizing this. The 
file Whiskey_unsuper_learn.ipynb contains the functions and parameter tuning I used for optimizing my t-SNE visualizations.
<br>
<img src="https://github.com/gspahlin/Whiskey_Analysis/blob/master/Figures/tSNE_alone.png" alt = "t-SNE example">
<br>
Fig. 1 - An example t-SNE visualization of the dimensionality reduced review features. The particular features used in the visualization above derive from counts of 
descriptive words by category (e.g. sweet, or fruity). This was the feature space I got the best results from.<br>

I used several different feature sets for my analysis, and every time I tried a new set of features, I used t-SNE to visualize my data before I did any clustering 
analysis. The following files are optimizations of t-SNE visualizations:
<br>
Whiskey_unsuper_learn.ipynb<br>
Whiskey_unsuper_learn_w_price.ipynb<br>
Whiskey_unsuper_learn_w_price_no_oak.ipynb<br>
Whiskey_unsuper_tsne5.ipynb<br>
Whiskey_unsuper_tsne6.ipynb<br>
Whiskey_unsuper_tsne7.ipynb<br>
