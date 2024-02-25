# Group Project for CSE151A

The exploratory data analysis conducted on the mushroom datasets provides insight into the distinguishing and classifying a specific mushroom's habitat. The primary dataset, with 173 observations across 23 features, including cap dimensions, colors, shape, surface, and more, reveals details about biodiversity. It does contain a number of null values which are concentrated in about five features. The secondary dataset expands the scope with thousands of observations for stem dimensions, ring type, and cap shape, alongside other features, and is devoid of null data. We performed introductory visual and statistical analysis, and saw a few patterns emerge, such as the distribution of cap diameters and color variations. These datasets' have the potential to reveal discriminative features critical for distinguishing digestibility. Through careful data preprocessing—including cleaning, normalizing, and feature extraction—we aim to move closer to accurately distinguishing between edible and poisonous mushrooms.

After we removed any Nan values, we alo conducted basic data cleaning such as one-hot encoding of our categorical features and output class. In addition to that, we used Chi-Squared tests and the SelectKBest library to determine the most significant features uin our dataset. In addition to that, we used Random Forests to determine the importance of various features, and from this we were able to determine the most significant features to impact the performance of the model, and to reduce the overall dimensionality of the model. We created multiple distibutions to determine how the classses are associated with the various features, such as seeing the distribution of the cap-shape values by the habitat of the various mushrooms. We also created a correlatioon matrix between the variosu features, to determine which of them are the most correlated and can affect each other. 


Below is a learning graph that compares the accuracy scores of the training set, the test set, and the cross validation set when using our model as we increase the set size to 35000. The graph shows that the accuracy score of the training set, the cross validation set, and the test set of our model are very close as we continue to increase the set size–in fact just thousandths away. This shows us that there isn't any overfitting. If there was overfitting, the scores would be wildly different. 
![image](./Learning_Curve_Graph.png)

Here is the full classification reports for the training set, validation set, and test set.
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

![iamge](./Confusion_Matrix.png)

The next two models we are thinking of using on this are random forest and k-nearest neighbors. We thought these were the best because random forest **INSERT REASON HERE.**
As for k-nearest neighbors, given that all of our habitat labels correspond to a number, **SOMEONE FOLLOW UP**

