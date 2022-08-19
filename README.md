# ml-razor

ml-razor is used to find an optimal point keeping you machine learning model performance high, using the fewest features. It does so by eliminating features based on their inter-correlation or their pre-defined importance. When using the inter-correlation eliminator, the correlations between all features are calculated. If their correlation is higher than a certain threshold, the one with the lowest feature importance is eliminated. When using the importance eliminator, features are one-by-one eliminated, sorted by their importance. In both situations, the model is iteratively retrained. Subsequently, a Kolmogorov-Smirnov analysis is used to find the optimal point where model performance is kept, while eliminating as much as possible features.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install ml-razor.

```bash
pip install ml-razor
```

## Usage
Suppose you have a dataset with too many continuous features. You have little domain knowledge, and therefore you want a data-driven approach for minimizing the amount of features while keeping model performance high.
```python
from ml_razor import Razor
from sklearn.datasets import load_breast_cancer
import ppscore as pps
import lightgbm as lgb


# get data
data, target = load_breast_cancer(return_X_y = True, as_frame = True)
data['target'] = target

# compute feature importances. this can be shapley values, importances from a random forest estimator for instance or
# some other measure. here we choose the predictive power score. the importances should be in the form of a dict,
# with all feature names as keys (str) and all importance values as values (float).
pps_results = pps.predictors(data, y='target')
pps_importances = {k: v for k, v in zip(pps_results['x'], pps_results['ppscore'])}

# define an estimator. This could be anything. for now we choose a simple lgbm classifier.
estimator = lgb.LGBMClassifier(max_depth=5)

# initiate the Razor. only the estimator parameter is not optional. cv = 10 means that for every correlation value
# between 1.00 and .25 (step size 0.01) the model is cross-validated with k=10. scoring='accuracy' for simplification
# purposes (in case of breast cancer this is probably not the right metric). p_alpha=.05 is the hypothesis testing
# threshold for the Kolmogorov-Smirnov analysis.
razor = Razor(estimator=estimator, cv=10, scoring='accuracy', lower_bound=.25, step=.01, method='correlation', p_alpha=.05)

# with 'shave' we fit the models for every correlation value between 1.00 and 0.25 with step size 0.01. this means,
# we compute the correlations between all features and when the correlation between feature_1 and feature_2 is
# greater than, let's say, 0.99 and the importance score of feature_2 is greater than feature_1, we drop feature_1.
# Then we proceed to correlation 0.98, and so on.
razor.shave(df=data, target='target', feature_importances=pps_importances)

# with plot, the results of the Kolmogorov-Smirnov analysis is shown. Or when setting plot_type to 'feature_impact', the
# feature impact and their corresponding importances are shown.
razor.plot(plot_type='ks_analysis')

# correlation_features is a list with the names of the features left after shaving with the razor. (i.e. optimal set of
# features for keeping model performance high with the least amount of features.)
correlation_features = razor.features_left

# correlation_importances is a dict, conserving only the predictive power scores (in this case) of the remaining
# features (correlation_features in this case).
correlation_importances = {k: v for k, v in pps_importances.items() if k in correlation_features}

# now we proceed to the next phase. we can fit the razor with another method ('importance'), using the correlation
# features as input. we start with all correlation_features and eliminate them one-by-one, beginning with the least
# important feature.
razor = Razor(estimator=estimator, scoring='accuracy', method='importance')
razor.shave(df=data, target='target', feature_importances=correlation_importances)
# in this particular case, it throws a warning saying that it didn't find an optimal point. this is due to the fact
# that there are only 3 features needed to keep the performance of the model high. with this little amount of data the
# Kolmogorov-Smirnov analysis has not enough evidence for rejecting the null-hypothesis. but this is beyond the scope
# of this introduction.

razor.plot(plot_type='feature_impact')

# the final_features is optimal set of features for keeping model performance high with the least amount of features.
final_features = razor.features_left
```
## License
[MIT](https://choosealicense.com/licenses/mit/)