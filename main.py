from ml_razor import Razor
from sklearn.datasets import load_breast_cancer
import ppscore as pps
import lightgbm as lgb

data, target = load_breast_cancer(return_X_y = True, as_frame = True)
data['target'] = target

pps_results = pps.predictors(data, y='target')
pps_importances = {k: v for k, v in zip(pps_results['x'], pps_results['ppscore'])}

estimator = lgb.LGBMClassifier(max_depth=5)

razor = Razor(estimator=estimator, cv=10, scoring='accuracy', lower_bound=.25, step=.01, method='correlation', p_alpha=.05)
razor.shave(df=data, target='target', feature_importances=pps_importances)
razor.plot(plot_type='ks_analysis')
correlation_features = razor.features_left
correlation_importances = {k: v for k, v in pps_importances.items() if k in correlation_features}

razor = Razor(estimator=estimator, scoring='accuracy', method='importance')
razor.shave(df=data, target='target', feature_importances=correlation_importances)
razor.plot(plot_type='feature_impact')
final_features = razor.features_left

