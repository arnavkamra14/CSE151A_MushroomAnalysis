# CSE151A Mushroom Analysis Project 
### Introduction 
Why chosen? why is it cool? General/Broader impact of having a good predictive mode. i.e. why is this important?

The choice to focus on mushroom analysis in this project stems from a blend of biology, data science, and biodiversity conservation. Mushrooms, with their diverse species, some edible and others potentially lethal, present a unique challenge in terms of classification and identification. The ability to accurately predict whether a mushroom is edible or poisonous through machine learning models not only has direct implications for food safety but also contributes to the broader field of biological classification and environmental science.
The significance of this project extends beyond the immediate practicality of mushroom classification. Within it lies its ability to demonstrate how data and computer science can be applied to solve real-world problems in fields outside of traditional technology sectors. By leveraging the vast biodiversity of mushrooms and the detailed data available on their physical characteristics, this project showcases the power of machine learning in enhancing our understanding of the natural world. A predictive model with high accuracy can significantly reduce the risk of mushroom poisoning incidents, which annually affect thousands of people worldwide, sometimes with fatal outcomes.
The broader impact of developing a reliable predictive model for mushroom classification extends beyond immediate public health concerns. It contributes to the fields of mycology and ecology by providing insights into mushroom distribution, diversity, and characteristics. This knowledge can aid in conservation efforts, help monitor ecological changes, and even support the discovery of new mushroom species or medicinal compounds.
In essence, this project is not only intriguing because of its application of advanced data analytics to an unconventional and challenging dataset but also because it stands at the convergence of technology and nature. It underscores the potential of machine learning to contribute significantly to various scientific domains, highlighting the importance of interdisciplinary research and the vast possibilities that open up when different fields of study collaborate. Through this mushroom analysis project, we aim to advance our understanding of the natural world, improve public health safety, and demonstrate the far-reaching applications of computer science.

### Figures
![image](./B-Cap_Shape_dist.png)
![image](./C-Cap_Shape_dist.png)
![image](./F-Cap_Shape_dist.png)
### Methods
##### Data Exploration 
Our initial approach involved performing exploratory data analysis (EDA) on the mushroom datasets to gain insights into the distinguishing characteristics that could help distinguish a mushroom's habitat and edibility. The primary dataset, with 173 observations across 23 features, including cap dimensions, colors, shape, surface, and more, reveals details about biodiversity. It does contain a number of null values which are concentrated in about five features. The secondary dataset provided a broader perspective with thousands of observations and included additional features such as stem dimensions, ring type, and cap shape, without any missing data. We performed introductory visual and statistical analysis and saw a few patterns emerge, such as the distribution of cap diameters and color variations. These datasets have the potential to reveal discriminative features critical for distinguishing digestibility. Through careful data preprocessing—including cleaning, normalizing, and feature extraction—we aim to move closer to accurately distinguishing between edible and poisonous mushrooms.
[View the EDA Notebook](https://github.com/arnavkamra14/CSE151A_MushroomAnalysis/blob/RandomForest/Mushroom_Classification_CSE151A.ipynb)
##### Preprocessing 
The preprocessing phase was crucial for preparing the data for modeling. This phase involved several steps: handling missing values, data cleaning, feature importance and selection, data balancing, visualization of distributions, and correlation analysis. We first removed instances with NaN values to ensure data integrity. Then we transformed our categorical features and output class using one-hot encoding to facilitate model processing. In addition to that, we utilized Chi-Squared tests and the SelectKBest library to identify the most significant features, reducing model complexity and focusing on relevant data. Our next step was to use Random Forests to determine the importance of various features. From this, we were able to determine the most significant features to impact the performance of the model and to reduce the overall dimensionality of the model. We created multiple distributions to determine how the classes are associated with the various features, such as seeing the distribution of the cap-shape values by the habitat of the various mushrooms. We also created a correlation matrix between the various features, to determine which of them are the most correlated and can affect each other.

We also implemented the Synthetic Minority Over-sampling Technique (SMOTE) to address imbalances in habitat distribution, particularly the overrepresentation of mushrooms found in woods(over 40,000), ensuring a uniform distribution across habitats. Oversampling resulted in a uniform distribution of the habitats in which the mushrooms were found. After oversampling, we analyzed cap-shape distributions across different habitats to understand feature-class relationships. The B-cap shape distribution, C-Cap shape distribution, and F-cap shape distribution can be seen above. There were vastly more mushrooms that had a stem width of 0 compared to a stem width of 1.

----------------------------------------------------------------------------------------
##### Model 1
Our first model was a logistic regression classifier, chosen for its simplicity and effectiveness in binary classification tasks. We evaluated the parameters of set size, regularization, and ross-validation. Below is a learning graph that compares the accuracy scores of the training set, the test set, and the cross-validation set when using our model as we increase the set size to 35000. The graph shows that the accuracy score of the training set, the cross-validation set, and the test set of our model are very close as we continue to increase the set size–in fact just thousandths away. This shows us that there isn't any overfitting. If there was overfitting, the scores would be wildly different. 
![image](./Learning_Curve_Graph.png)

The next two models we are considering using on this are random forest classification and k-nearest neighbors. We thought these were the best because random forest classification is likely to output an accurate prediction for classification problems because it uses multiple decision trees with various subsets of the dataset. The problem with random forest classification algorithms is that they tend to overfit to the training dataset, so we will need to be careful when training our model.  As for k-nearest neighbors, given that all of our habitat labels correspond to a specific group, it will be easy to use a clustering algorithm to classify the various mushrooms. Clustering tends to be a very versatile method for classification and k-nearest neighbors is an efficient clustering algorithm that we can test with various parameters. 

----------------------------------------------------------------------------------------
##### Model 2
After assessing the performance of the logistic regression model, we explored K-Nearest Neighbors (KNN) for its simplicity and efficacy in classification based on feature similarity. As parameters, we considered the number of neighbors, distance metric, and weight function.

Below are the learning curve, confusion matrix, and classification report. 

![image](./Learning_Curve_KNN.png)
![image](./Confusion_Matrices_KNN.png)
![image](./Classification_Report_KNN.png)

From these visuals, we can tell that as we increase the training set size, the accuracy of both the training set and the cross-validation set increase, surpassing the test score. Overall, when compared to the first model, the accuracy increased amongst all sets. 

The hyperparameter tuning that we did involved using cross-validation with GridSearchCV to test the number of nearest neighbors that yield the highest accuracy. We used values [1,3,5,7,9] for the number of nearest neighbors, and we found that 9 yielded the highest accuracy of 0.74.

Something notable about the hyperparameter tuning is that it resulted in the accuracy scores of our logistic regression model decreasing (as seen below). However, our K-nearest neighbors model was still more accurate than our first model with the initial data used.

![image](./Learning_Curve_Log_Reg.png)

When looking at our second model, we decided to continue with our original plan of using random forest classification to see if we can increase accuracy even further. This is because random forest classification tends to be more accurate than KNN because of its nature as an ensemble model and its generally accurate performance, relative to K-nearest neighbors. 
##### Model 3

``` from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300], 
    'max_depth': [None, 10, 20, 30],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4]     
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score achieved: {grid_search.best_score_}")

# Use the best estimator to make predictions on the test set
best_estimator = grid_search.best_estimator_
y_pred = best_estimator.predict(X_test)

from sklearn.metrics import accuracy_score
print(f"Test set accuracy: {accuracy_score(y_test, y_pred)}") ```
### Results
##### Data Exploration
##### Preprocessing
##### Model 1

Here are the full classification reports for the training set, validation set, and test set on this model.
![image](./Classification_Report.png)

Below is a confusion matrix of the predicted habitat and the true habitat. Each of the letters corresponds to the following:
- w = waste
- u = urban
- p = paths
- m = meadows
- l = leaves
- h = heaths
- g = grasses
- d = woods

![image](./Confusion_Matrix.png)

##### Model 2
We are happy with the results with our 2nd model, a K-nearest neighbors model using the 5 nearest neighbors. There was no overfitting, the accuracy score kept increasing with an increased training set size, and it did better than our first model. 

To improve upon this model, we can use further test out different amounts of nearest neighbors and see what value is most optimal by comparing the performance of the model on our test data set after verifying its performance on our training data set. 
##### Model 3


### Discussion
Discussion section: This is where you will discuss the why, and your interpretation and your though process from beginning to end. This will mimic the sections you have created in your methods section as well as new sections you feel you need to create. You can also discuss how believable your results are at each step. You can discuss any short comings. It's ok to criticize as this shows your intellectual merit, as to how you are thinking about things scientifically and how you are able to correctly scrutinize things and find short comings. In science we never really find the perfect solution, especially since we know something will probably come up int he future (i.e. donkeys) and mess everything up. If you do it's probably a unicorn or the data and model you chose are just perfect for each other!

### Conclusion
This is where you do a mind dump on your opinions and possible future directions. Basically what you wish you could have done differently. Here you close with final thoughts
### Collaboration
 statement of contribution by each member. This will be taken into consideration when making the final grade for each member in the group. Did you work as a team? was there a team leader? project manager? coding? writer? etc. Please be truthful about this as this will determine individual grades in participation. There is no job that is better than the other. If you did no code but did the entire write up and gave feedback during the steps and collaborated then you would still get full credit. If you only coded but gave feedback on the write up and other things, then you still get full credit. If you managed everyone and the deadlines and setup meetings and communicated with teaching staff only then you get full credit. Every role is important as long as you collaborated and were integral to the completion of the project. If the person did nothing. they risk getting a big fat 0. Just like in any job, if you did nothing, you have the risk of getting fired. Teamwork is one of the most important qualities in industry and academia!!!
Start with Name: Title: Contribution. If the person contributed nothing then just put in writing: Did not participate in the project.
