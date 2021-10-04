# Whiskey_Analysis
Unsupervised learning and analysis with my whiskey database
<br><br>
The goal of this analysis was to understand what kinds of whiskeys are available in the market - in order to provide recommendations. My underlying hypothesis is that 
whiskeys will cluster by price, flavor characters, and abv in ways that are not captured by categories such as 'single malt scotch' or 'bourbon/tenessee' which are 
largely geographical in nature. I am using the t-SNE algorythm in Scikit-Learn to visualize the structure of the data in my feature space. The feature spaces I used 
in
this analysis all come from the review language present on WhiskeyAdvocate.com. They were constructed using word counts of discriptive words in the reviews. After 
visualization of the data, the next step is to use a clustering algorythm to predict clusters in the data. I will use my best t-SNE parameters to visualize the 
clusters, 
and basic data analysis to learn the characteristics of the clusters and find representatives. 
<br><br>
Step 1 - Optimizing the visualization
<br>
In this study I was interested in using unsupervised clustering methods to identify similar whiskeys. I wanted some kind of visualization technique, however, in order
to see what my clustering algorythm was actually doing. I chose t-SNE for this, which is a dimensionality reduction technique that relies on a distance metric to 
determine where points should fall in relation to one another in a 2 dimensional space. The first code I commited to this repository was in the service of optimizing 
this. The file Whiskey_unsuper_learn.ipynb contains the functions and parameter tuning I used for optimizing my t-SNE visualizations.
<br>
<img src="https://github.com/gspahlin/Whiskey_Analysis/blob/master/Figures/tSNE_alone.png" alt = "t-SNE example">
<br>
Fig. 1 - An example t-SNE visualization of the dimensionality reduced review features. The particular features used in the visualization above derive from counts of 
descriptive words by category (e.g. sweet, or fruity). This was the feature space I got the best results from.<br>
<br><br>
I used several different feature sets for my analysis, and every time I tried a new set of features, I used t-SNE to visualize my data before I did any clustering 
analysis. The following files are optimizations of t-SNE visualizations:
<br>
Whiskey_unsuper_learn.ipynb<br>
Whiskey_unsuper_learn_w_price.ipynb<br>
Whiskey_unsuper_learn_w_price_no_oak.ipynb<br>
Whiskey_unsuper_tsne5.ipynb<br>
Whiskey_unsuper_tsne6.ipynb<br>
Whiskey_unsuper_tsne7.ipynb<br>
<br>
Step 2 - clustering analysis to identify similarities in review language
<br>
Once I had a method for visualizing the data, I could move on to trying to use a clustering method for identifying similar review language in whiskeys. The assumption 
here is that reviews that use similar descriptive language will correspond to whiskies with similar flavor and sensory characteristics. The clustering method I used 
for clustering was Density-Based Spatial Clustering of Applications with Noise (DBSCAN). I selected DBSCAN because it functions by identifying clusters based on 
density. This makes it possible for DBSCAN to identify clusters with unusual shapes. The parameters of clustering, make a big difference for how similar the review 
language has to be to match, and worked differently based on the features used. The features I used changed througout the study - first I started with a list of about 
30 of the most common descriptive words as features. I then reduced the feature space to a set of categories (e.g. sweet, wood, fruity) and enumerated the number of 
words that fit into certain flavor categories. Finally I tried a version of my features where the number of words in particular flavor categories were considered as 
ratios of the total number of words in the review. Of these, the categorical features that were not normalized appeard to work the best. 
<br>
In order visualize the clusters, I added clustering to my t-SNE visualizations, where individual clusters. Depending on the parameters used, you can dramatically 
chaneg the number of clusters found by the algorythm. Below is a an early clusering pattern that I found in my analyses. 
<br>
<img src= "https://github.com/gspahlin/Whiskey_Analysis/blob/master/Figures/suboptimal_clusters.png" alt = "clustering results">
<br>
Fig. 2 - clustering results with a small number of clusters. 

Initially I thought this clustering pattern looked promising. In the figure above, you can see that the whiskeys have been placed in 7 categories - the 8th category 
(labled -1) is a noise category, where uncategorized whiskies are placed. To figure out how the whiskies are being categorized in this scheme, I used GroupBy 
statistics. Doing this clearly shows that the whiskeys are primarily being categorized by how many sweet descriptors appear in the review.
<br>
<img src= "https://github.com/gspahlin/Whiskey_Analysis/blob/master/Figures/Clustering_results_too_coarse.jpg" alt = "clustering results">
<br>
Fig. 3 - Overly course DBSCAN criteria lead to whiskeys being grouped almost completely by how sweet they are. 
<br><br>
Figure 3 provides some inight into how the DBSCAN algorythm is clusering these whiskies. In the course method shown above, the critera for being in a particular 
category is the number of times a descrpitions present in the review falls into the "sweet" category (e.g. sweet, sweetness, honey, or sugar). So a whiskey that was 
described as "sweet" twice, would be placed in the same category a one that described a "honey" flavor, and went on to describe the "sweetness" of the whiskey. In 
this case 372 whiskeys were classified as 'noise', which is about 6% of the whiskeys. In order to get more useful categorizations. In order to maximize the number of 
groups minimized epsilon (the maximum distance to the next point in the cluster) and "min_samples", the number points that need to be less than epsilon away in order 
for a point to be in the center of a cluster. The result of this was a clustering protocol with 532 distinct clusters. The number of whiskeys in cluster -1 was 731 
whiskies, or about 12% of whiskies. In spite of the uncategorized figure increasing, the groups in this method are far more targeted. See below for an example.
<br>
<img src= "https://github.com/gspahlin/Whiskey_Analysis/blob/master/Figures/HS_clusters.png" alt = "clustering results optimized">
<br>
Fig. 4 - A DBSCAN clustring algorythm that results in 532 categories. There are not 532 colors so some of the groups are redundant. 
<br><br>
When the criteria for forming a group are lowered to the highest degree, the groups get much smaller, and the criteria for being in the same group gets higher. In 
general most of these whiskies are the same in all but one or two of the available features. 
<br>
<img src= "https://github.com/gspahlin/Whiskey_Analysis/blob/master/Figures/Representative%20cluster.jpg" alt = "clustering results optimized">
<br>
Fig. 5 - The optimized clustering algorythm produces small groups with high level of similarities of descriptive words in the reviews. In the above example the 
whiskeys in the same group have agreement in all of the features except for "fruity words".  
<br>
<br>
Several different feature spaces were tried at this level. The best results came from a feature that enumerated instances of descriptive language in a set of 
different categories. This worked better than a large set of descriptive words, and better than the descriptive words when they were normalized to the number of words 
in the review. There may be room for improvement in this scheme, but at the moment I feel that I've found a maximum in the particular approach I'm using.
<br>
Currently the clustering file that is most important is:<br>
<br>
<b>Whiskey_unsuper_learn_cluster3.ipynb</b>
<br>
<br>
Other clustering result files include:
<br><br>
Whiskey_unsuper_learn_cluster1.ipynb<br>
Whiskey_unsuper_learn_cluster2.ipyn<br>
Whiskey_unsuper_learn_cluster3.ipynb<br>
<br><br>
Finally the files for annotation of this read me are present in the figures file. In the future I may decide to revisit the features and this analysis, but at present 
I plan to advance to the next part of my project with the current clustering data for providing recommendations. The clustering results mentioned above were loaded 
into my updated research database, along with my various revises feature sets. 
<br><br>
<img src= "https://raw.githubusercontent.com/gspahlin/WhiskyAdvocate_ETL/master/ERD_and_SQL/Whiskey_ERD_1.jpg" alt = "updated ERD">
<br>
Fig. 6 - The optimized clustering algorythm produces small groups with high level of similarities of descriptive words in the reviews. In the above example the 
whiskeys in the same group have agreement in all of the features except for "fruity words".  
<br><br>
Stay tuned for a web application to provide 
recommendations! 
 <br><br>
 Gregory W. Spahlinger   
 gspahlin@gmail.com    
 <a href = 'https://www.linkedin.com/in/gregory-spahlinger/'>LinkedIn</a>
