import unittest

from ml_razor import Razor

from os import cpu_count
import numpy as np
import pandas as pd

from scipy.stats import random_correlation
from scipy.linalg import cholesky

import ppscore as pps
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

def create_correlated_regression_data(correlation_level, importance_type, estimation, random_state):
    """
    correlation_level : str
        Possible values {'high', 'low', 'none'}.
    importance_type : str
        Possible values {'pps', 'rf'}.
    estimation : str
        Possible values {'linear_regression', 'light_gradient_boost'.
    random_state : int
        Must be > 0.
    """
    N = 1000
    n_features = 101
    
    if correlation_level != 'none':
        correlation_level_int = 10 if correlation_level == 'high' else 3
        eig = [0.1] * (n_features - correlation_level_int)
        eig = np.append(eig, [n_features / correlation_level_int] * correlation_level_int)
        rest = n_features - np.sum(eig)
        eig[-1] += rest
        cm = random_correlation.rvs(eig, random_state=random_state)
        upper_chol = cholesky(cm)
        np.random.seed(random_state)
        rnd = np.random.normal(0.0, 1.0, size=(N, n_features))
        X = rnd @ upper_chol
    else:  # correlation_level == 'none'
        np.random.seed(random_state)
        X = np.random.normal(0.0, 1.0, size=(N, n_features))

    y = X.T[-1]
    X = X.T[:-1].T
    df = pd.DataFrame(X, columns=['feature_{}'.format(i) for i in range(n_features - 1)])
    df['target'] = y

    if importance_type == 'pps':
        pps_results = pps.predictors(df, y='target')
        importances = {k: v for k, v in zip(pps_results['x'], pps_results['ppscore'])}
    else:  # importance_type == 'rf'
        forest = RandomForestRegressor(random_state=random_state)
        forest.fit(X, y)
        fi = forest.feature_importances_
        importances = {k: v for k, v in zip(df.columns, fi)}

    estimator = LinearRegression() if estimation == 'linear_regression' else lgb.LGBMRegressor(max_depth=5, n_jobs=cpu_count() - 1)

    return df, importances, estimator


class RazorTestCase(unittest.TestCase):

    def _fit_correlation(self, df, importances, estimator):
        self.razor = Razor(estimator=estimator, random_state=self.__random_state, verbose=0)
        self.razor.shave(df=df, target='target', feature_importances=importances)
        correlation_features = self.razor.features_left
        correlation_importances = {k: v for k, v in importances.items() if k in correlation_features}

        return correlation_features, correlation_importances

    def _fit_importance(self, df, importances, estimator):
        self.razor = Razor(estimator=estimator, method='importance', random_state=self.__random_state, verbose=0)
        self.razor.shave(df=df, target='target', feature_importances=importances)
        final_features = self.razor.features_left

        return final_features

    def _correlation_assertions(self, df, correlation_features):
        self.assertTrue(all(np.array([f in df.columns for f in correlation_features])))
        self.assertGreaterEqual(self.razor.optimal_point, .25)
        self.assertLessEqual(self.razor.optimal_point, 1)

    def _importance_assertions(self, correlation_features, final_features):
        self.assertTrue(all(np.array([f in correlation_features for f in final_features])))
        self.assertGreaterEqual(self.razor.optimal_point, 1)
        self.assertLessEqual(self.razor.optimal_point, len(correlation_features))

    def setUp(self):
        self.__random_state = 12
        self.razor = None

    def test_high_cor_pps_linreg(self):
        df, importances, estimator = create_correlated_regression_data('high', 'pps', 'linear_regression', self.__random_state)

        correlation_features, correlation_importances = self._fit_correlation(df, importances, estimator)
        self._correlation_assertions(df, correlation_features)

        final_features = self._fit_importance(df, correlation_importances, estimator)
        self._importance_assertions(correlation_features, final_features)

    def test_high_cor_rf_linreg(self):
        df, importances, estimator = create_correlated_regression_data('high', 'rf', 'linear_regression', self.__random_state)

        correlation_features, correlation_importances = self._fit_correlation(df, importances, estimator)
        self._correlation_assertions(df, correlation_features)

        final_features = self._fit_importance(df, correlation_importances, estimator)
        self._importance_assertions(correlation_features, final_features)

    def test_low_cor_pps_linreg(self):
        df, importances, estimator = create_correlated_regression_data('low', 'pps', 'linear_regression', self.__random_state)

        correlation_features, correlation_importances = self._fit_correlation(df, importances, estimator)
        self._correlation_assertions(df, correlation_features)

        final_features = self._fit_importance(df, correlation_importances, estimator)
        self._importance_assertions(correlation_features, final_features)

    def test_low_cor_pps_lgbm(self):
        df, importances, estimator = create_correlated_regression_data('low', 'pps', 'light_gradient_boost', self.__random_state)

        correlation_features, correlation_importances = self._fit_correlation(df, importances, estimator)
        self._correlation_assertions(df, correlation_features)

        final_features = self._fit_importance(df, correlation_importances, estimator)
        self._importance_assertions(correlation_features, final_features)

    def test_low_cor_rf_linreg(self):
        df, importances, estimator = create_correlated_regression_data('low', 'rf', 'linear_regression', self.__random_state)

        correlation_features, correlation_importances = self._fit_correlation(df, importances, estimator)
        self._correlation_assertions(df, correlation_features)

        final_features = self._fit_importance(df, correlation_importances, estimator)
        self._importance_assertions(correlation_features, final_features)

    def test_no_cor_pps_linreg(self):
        df, importances, estimator = create_correlated_regression_data('none', 'pps', 'linear_regression')

        with self.assertWarns(UserWarning):
            correlation_features, correlation_importances = self._fit_correlation(df, importances, estimator)
        self._correlation_assertions(df, correlation_features)

        final_features = self._fit_importance(df, correlation_importances, estimator)
        self._importance_assertions(correlation_features, final_features)


if __name__ == '__main__':
    unittest.main()
