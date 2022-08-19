import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.base import clone
import warnings


class Razor:
    """
    A class used to find an optimal point keeping model performance high, using the fewest features. It does so by
    eliminating features based on their inter-correlation or their pre-defined importance. When using the
    inter-correlation eliminator, the correlations between all features are calculated. If their correlation is
    higher than a certain threshold, the one with the lowest feature importance is eliminated. When using the
    importance eliminator, features are one-by-one eliminated, sorted by their importance. In both situations,
    the model is iteratively retrained. Subsequently, a Kolmogorov-Smirnov analysis is used to find the optimal point
    where model performance is kept, while eliminating as much as possible features.

    ...

    Attributes
    ----------
    optimal_point : float or int
        In case of 'correlation' method a float representing the optimal inter-correlation threshold to eliminate one
        of the correlating features. In case of 'importance' method an int representing the optimal importance value
        from where features are eliminated.
    deterioration_point : float or int
        In case of 'correlation' method a float representing the first correlation value where model performance starts
        to decline. In case of 'importance' method an int representing the first importance value where model
        performance starts to decline.
    features_left : list
        Optimal set of features after eliminating by correlation or importance.

    Methods
    -------
    shave(df, features, target, feature_importances)
        The main algorithm used to find an optimal
        point keeping model performance high, using the fewest features. It does so by eliminating features based on
        their inter-correlation or their pre-defined importance. When using the inter-correlation eliminator,
        the correlations between all features are calculated. If their correlation is higher than a certain threshold,
        the one with the lowest feature importance is eliminated. When using the importance eliminator, features are
        one-by-one eliminated, sorted by their importance. In both situations, the model is then retrained.
        Subsequently, a Kolmogorov-Smirnov analysis is used to find the optimal point where model performance is kept,
        while eliminating as much as possible features.
    plot(plot_type='ks_analysis')
        Two types of visualizations to explain how the optimal point was found.
    """

    def __init__(self, estimator,
                 cv=10,
                 scoring='neg_root_mean_squared_error',
                 lower_bound=.25,
                 step=.01,
                 method='correlation',
                 p_alpha=.05,
                 random_state=None,
                 verbose=1):

        """

        :param estimator: The object to use to fit the data.
        :type estimator: estimator object
        :param cv: Determines the cross-validation splitting strategy. Must be between [5, 100]. Default is 10.
        :type cv: int, optional
        :param scoring: Possible values of the scoring method are listed here:
                        https://scikit-learn.org/stable/modules/model_evaluation.html.
        :type scoring: str, optional
        :param lower_bound: When method = 'correlation', this is the end value of the search space (beginning at 1.0)
                            of all correlation values, type must be float. When method = 'importance', this is the
                            minimal number of features left after eliminating, type must be int. Must be between
                            [0, 0.80] in case of method = 'correlation' and > 0 in case of 'importance'. Default is
                            0.25 or 1.
        :type lower_bound: float or int, optional
        :param step: When method = 'correlation', this is the step size of the search space of all correlation values.
                     Must be between (0, 0.10]. Default is 0.01.
        :type step: float, optional
        :param method: Method to use. Default is 'correlation'.
        :type method: {'correlation', 'importance'}, optional
        :param p_alpha: Threshold for the p-value of the Kolmogorov-Smirnov analysis.  Must be between [0.01, .25].
                        Default is 0.05.
        :type p_alpha: float, optional
        :param random_state: Determines random_state for cross validation part of the algorithm.
        :type random_state: int, optional
        :param verbose: If > 0, it shows the progress while fitting the models.
        :type verbose: int, optional
        """

        # parameters
        self.__estimator = estimator
        self.__cv = cv
        self.__scoring = scoring
        if (method == 'importance') & (lower_bound == .25):
            self.__lower_bound = 1
        else:
            self.__lower_bound = lower_bound
        self.__step = step
        self.__method = method
        self.__p_alpha = p_alpha
        self.__random_state = random_state
        self.__verbose = verbose
        self.__feature_importances = None
        self.__features = None

        # attributes
        self.optimal_point = None
        self.deterioration_point = None
        self.features_left = None

        # variables
        self.__df_models = None
        self.__df_ks_analyses = None
        self.__df_ks_analyses_summarized = None
        self.__drop_conditions = None
        self.__n_include = None
        self.__model_cv_scores = None
        self.__dropped_features = None
        self.__weighted_pvalue = None
        self.__df_deterioration_ks_analysis = None

    def _fit_model(self, estimator, X, y, cv, RANDOM_STATE):
        estimator = clone(estimator)
        kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(estimator, X, y, cv=kf, scoring=self.__scoring)

        return scores

    def _create_initial_data(self, df, target, feature_importances):
        fi = {k: v for k, v in reversed(sorted(feature_importances.items(), key=lambda item: item[1]))}
        self.__feature_importances = [v for k, v in fi.items()]
        self.__features = [k for k, v in fi.items()]
        Xdf = df.loc[:, self.__features]
        y = df.loc[:, target].values.ravel()
        cols = Xdf.columns

        return Xdf, y, cols

    def _create_df_models_correlation(self, df, target, Xdf, y, cols, random_counter, metrics):
        cors = np.round(np.linspace(1, self.__lower_bound, num=1 + int((1 - self.__lower_bound) / self.__step)), 2)
        if self.__random_state is not None:
            np.random.seed(self.__random_state)
        RANDOM_STATES = np.random.randint(low=1, high=1000, size=len(cors))
        df_cors = Xdf.corr()
        df_cors = df_cors.unstack().drop_duplicates()
        for f in self.__features:
            if (f, f) in df_cors.index:
                df_cors.drop((f, f), axis=0, inplace=True)

        for c in cors:
            feature_drop_random = set([ii[1] for ii in df_cors[df_cors.abs() > c].index])
            feature_drop = [f for f in self.__features if f in feature_drop_random]
            leftover = list(set(self.__features) - feature_drop_random)
            leftover = [bf for bf in self.__features if bf in leftover]
            dfi = df.loc[:, leftover + [target]]
            X = dfi.loc[:, leftover].values

            RANDOM_STATE = RANDOM_STATES[random_counter]
            scores = self._fit_model(self.__estimator, X, y, self.__cv, RANDOM_STATE)
            random_counter += 1

            self.__model_cv_scores.append(scores)
            self.__dropped_features.append(feature_drop)
            metrics['drop_condition'].extend([c] * self.__cv)
            metrics['features_left'].extend([leftover] * self.__cv)
            metrics['n_features'].extend([len(leftover)] * self.__cv)
            metrics[self.__scoring].extend(scores)
            if self.__verbose > 0:
                print('Inter-correlation: {}, progress: {}%'.format(c,
                                                                    round(((list(cors).index(c) + 1) / len(cors)) * 100,
                                                                          1)), end='\r')

        self.__df_models = pd.DataFrame(metrics)

    def _create_df_models_importance(self, df, target, y, random_counter, metrics):
        max_features_drop = len(self.__features) - self.__lower_bound
        max_features_drop += 1
        if self.__random_state is not None:
            np.random.seed(self.__random_state)
        RANDOM_STATES = np.random.randint(low=1, high=1000, size=max_features_drop)
        for i in range(max_features_drop):
            if i == 0:
                feature_drop = ''
                leftover = self.__features[:]
            else:
                feature_drop = self.__features[-i]
                leftover = self.__features[:(-i)]
            dfi = df.loc[:, leftover + [target]]
            X = dfi.loc[:, leftover].values

            RANDOM_STATE = RANDOM_STATES[random_counter]
            scores = self._fit_model(self.__estimator, X, y, self.__cv, RANDOM_STATE)
            random_counter += 1

            k = len(leftover)
            self.__model_cv_scores.append(scores)
            self.__dropped_features.append(feature_drop)
            metrics['drop_condition'].extend([k] * self.__cv)
            metrics['features_left'].extend([leftover] * self.__cv)
            metrics['n_features'].extend([k] * self.__cv)
            metrics[self.__scoring].extend(scores)
            if self.__verbose > 0:
                print('{} features left, progress: {}%'.format(k, round(((i + 1) / max_features_drop) * 100, 1)),
                      end='\r')

        self.__df_models = pd.DataFrame(metrics)

    def _create_batches(self, outbounds, scope):
        batch0_upper_border = outbounds
        batch1_lower_border = outbounds
        batch1_upper_border = outbounds + scope

        return batch0_upper_border, batch1_lower_border, batch1_upper_border

    def _create_df_ks_analyses(self):
        d = {'outbounds': [],
             'scope': [],
             'new_condition': [],
             'statistic': [],
             'pvalue': [],
             'batch0_mean': [],
             'batch1_mean': []}

        self.__n_include = 5
        self.__drop_conditions = self.__df_models['drop_condition'].unique()
        l = self.__df_models.shape[0]

        outbounds_start = self.__cv
        outbounds_end = int(l)
        scope_start = self.__cv
        scope_end = (self.__n_include * self.__cv) + self.__cv

        for outbounds in range(outbounds_start, outbounds_end, self.__cv):
            for scope in range(scope_start, scope_end, self.__cv):
                batch0_upper_border, batch1_lower_border, batch1_upper_border = self._create_batches(outbounds, scope)

                if batch1_upper_border < l:
                    batch0 = self.__df_models.loc[:batch0_upper_border - 1, self.__scoring]
                    batch1 = self.__df_models.loc[batch1_lower_border:batch1_upper_border - 1, self.__scoring]
                    new_condition = self.__df_models.loc[batch1_lower_border:batch1_upper_border - 1,
                                    'drop_condition'].iloc[0]

                    ks_results = ks_2samp(batch0, batch1)

                    d['outbounds'].append(outbounds)
                    d['scope'].append(scope)
                    d['new_condition'].append(new_condition)
                    d['statistic'].append(ks_results.statistic)
                    d['pvalue'].append(ks_results.pvalue)
                    d['batch0_mean'].append(np.mean(batch0))
                    d['batch1_mean'].append(np.mean(batch1))

        self.__df_ks_analyses = pd.DataFrame(d).sort_values(by=['new_condition', 'scope'], ascending=[False, True])

    def _create_df_ks_analyses_summarized(self):
        self.__df_ks_analyses_summarized = self.__df_ks_analyses.groupby('new_condition')[
            ['pvalue', 'batch0_mean', 'batch1_mean']].mean().sort_index(ascending=False)
        self.__df_ks_analyses_summarized['batch1_larger_than_batch0'] = self.__df_ks_analyses_summarized[
                                                                            'batch1_mean'] > \
                                                                        self.__df_ks_analyses_summarized['batch0_mean']
        self.__df_ks_analyses_summarized = self.__df_ks_analyses_summarized.drop(['batch0_mean', 'batch1_mean'], axis=1)

    def _create_optimal_deterioration_points(self):
        if any(self.__df_ks_analyses_summarized['pvalue'].values.flatten() < self.__p_alpha):
            p = 1
            i = -1
            larger = True
            while (p >= self.__p_alpha) | larger:
                i += 1
                if i >= self.__df_ks_analyses_summarized.shape[0]:
                    warnings.warn('There is no evidence for model deterioration with these input parameters. '
                                  'Try to extend the scope.')
                    i -= 1
                    break
                p = self.__df_ks_analyses_summarized['pvalue'].iloc[i]
                larger = self.__df_ks_analyses_summarized['batch1_larger_than_batch0'].iloc[i]

            self.optimal_point = self.__df_ks_analyses_summarized.index[i - 1]
            self.deterioration_point = self.__df_ks_analyses_summarized.index[i]
        else:
            self.optimal_point = self.__df_ks_analyses_summarized.index[-2]
            self.deterioration_point = self.__df_ks_analyses_summarized.index[-1]
            warnings.warn('There were no pvalues < {}, which means there is no evidence for model deterioration. '
                          'Plot the results for more insights.'.format(self.__p_alpha))

    def _create_features_left(self):
        self.features_left = \
            self.__df_models.loc[self.__df_models['drop_condition'] == self.optimal_point, 'features_left'].iloc[0]

    def _plot_ks_analysis(self):
        fig, ax = plt.subplots(2, self.__n_include, figsize=(4 * self.__n_include, 8), tight_layout=True)
        sr = self.__df_ks_analyses_summarized.reset_index()
        self.__weighted_pvalue = round(sr.loc[sr['new_condition'] == self.deterioration_point, 'pvalue'].values[0], 4)
        self.__df_deterioration_ks_analysis = self.__df_ks_analyses.query(
            'new_condition == {}'.format(self.deterioration_point)).reset_index(drop=True)

        if self.__method == 'correlation':
            main_title = 'Showing the decline in model performance for dropping one of two features ' \
                         'with their inter-correlation.\nThe orange dots represent the first models showing ' \
                         'performance decline.\nAdding a model where features are dropped with an ' \
                         'inter-correlation of <= {}, the KS analysis gives p = {}' \
                         '.'.format(self.deterioration_point, self.__weighted_pvalue)
        else:
            main_title = 'Showing the decline in model performance for dropping n features.\nThe orange dots ' \
                         'represent the first models showing performance decline.\nAdding a model where {} ' \
                         'features are left, the KS analysis gives p = {}' \
                         '.'.format(self.deterioration_point, self.__weighted_pvalue)

        fig.suptitle(main_title)

        fr_range = self.__df_deterioration_ks_analysis.index
        for i in fr_range:
            outbounds = self.__df_deterioration_ks_analysis.loc[i, 'outbounds']
            scope = self.__df_deterioration_ks_analysis.loc[i, 'scope']
            ks_statistic = round(self.__df_deterioration_ks_analysis.loc[i, 'statistic'], 4)
            ks_pvalue = round(self.__df_deterioration_ks_analysis.loc[i, 'pvalue'], 4)

            batch0_upper_border, batch1_lower_border, batch1_upper_border = self._create_batches(outbounds, scope)

            metric_batch0 = self.__df_models.loc[:batch0_upper_border - 1, self.__scoring]
            metric_batch1 = self.__df_models.loc[batch1_lower_border:batch1_upper_border - 1, self.__scoring]
            metric_batch2 = self.__df_models.loc[batch1_upper_border:, self.__scoring]

            method_batch0 = self.__df_models.loc[:batch0_upper_border - 1, 'drop_condition']
            method_batch1 = self.__df_models.loc[batch1_lower_border:batch1_upper_border - 1, 'drop_condition']
            method_batch2 = self.__df_models.loc[batch1_upper_border:, 'drop_condition']

            ax[0, i].invert_xaxis()
            ax[0, i].scatter(method_batch0, metric_batch0)
            ax[0, i].scatter(method_batch1,
                             metric_batch1,
                             color='orange')
            ax[0, i].scatter(method_batch2, metric_batch2, color='black',
                             alpha=.1)
            xlab = 'Correlation' if self.__method == 'correlation' else 'N features left'
            ax[0, i].set_xlabel(xlab)
            ax[0, i].set_ylabel(self.__scoring)

            ax[1, i].hist(metric_batch0,
                          density=True, weights=np.ones_like(metric_batch0) / len(metric_batch0))
            ax[1, i].hist(metric_batch1, color='orange',
                          density=True, weights=np.ones_like(metric_batch1) / len(metric_batch1))
            ax[1, i].set_title('{} Distributions\nKolmogorov-Smirnov results\n'
                               'statistic: {}\np-value: {}'.format(self.__scoring, ks_statistic, ks_pvalue))

    def _plot_feature_impact(self):
        col0 = 'black'
        col1 = 'purple'
        col2 = 'red'

        condition0 = self.optimal_point
        condition0_idx = np.where(self.__drop_conditions == condition0)[0][0]
        condition1 = self.__drop_conditions[condition0_idx + 1]
        vline = np.mean([condition0, condition1])
        x_axis = self.__drop_conditions

        fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        plt.subplots_adjust(hspace=.0)

        ax[0].plot(x_axis, self.__model_cv_scores, color=col0, marker='.', linestyle='None', label=self.__scoring)
        ax[0].set_ylabel(self.__scoring)
        ax[0].axvline(vline, color=col2, alpha=.3, linestyle='dashed')
        ax[0].invert_xaxis()
        if all(np.array(self.__feature_importances) == 0):
            ax[1].set_ylim([0, 1])

        if self.__method == 'correlation':
            y1_n_removed = [len(dropf) for dropf in self.__dropped_features]
            y1_imp_removed = [np.sum([self.__feature_importances[j]
                                      for j in [self.__features.index(i)
                                                for i in self.__dropped_features[k]]])
                              for k in range(len(self.__dropped_features))]
            bar_width = .8 * np.mean(np.diff(x_axis))
            text_align = min(y1_imp_removed)

            ax[1].bar(x=x_axis, height=y1_imp_removed, color=col1, alpha=.3, label='feature importance',
                      width=bar_width)
            for n, coord in zip(y1_n_removed, x_axis):
                ax[1].text(coord, text_align, '{} / {} features dropped'.format(n, len(self.__features)),
                           rotation=90, color=col1, alpha=.7, verticalalignment='bottom',
                           horizontalalignment='center')
            ax[1].set_ylabel('Cumulative importance of dropped features\n(individual importances may be negative)')
            ax[1].set_xlabel('Correlation')
            ax[1].axvline(vline, color=col2, alpha=.3, linestyle='dashed')
            main_title = 'When dropping features with an inter-correlation of' \
                         ' <= {}, model performance declines.'.format(condition1)
        else:
            y1_imp = [0] + [self.__feature_importances[self.__features.index(self.__dropped_features[i])]
                            for i in range(1, len(self.__dropped_features))]
            text_align = min(y1_imp)
            ax[1].bar(x=x_axis, height=y1_imp, color=col1, alpha=.3, label='feature importance')
            for name, coord in zip(self.__dropped_features, x_axis):
                ax[1].text(coord, text_align, name, rotation=90, color=col1, alpha=.7, verticalalignment='bottom',
                           horizontalalignment='center')
            ax[1].set_ylabel('Importance of dropped feature\n(importances may be negative)')
            ax[1].set_xlabel('N features left')
            ax[1].axvline(vline, color=col2, alpha=.3, linestyle='dashed')
            main_title = 'With <= {} features left, model performance declines.'.format(
                self.deterioration_point)

        fig.suptitle(main_title)

    def shave(self, df, target, feature_importances):
        """
        The main algorithm used to find an optimal
        point keeping model performance high, using the fewest features. It does so by eliminating features based on
        their inter-correlation or their pre-defined importance. When using the inter-correlation eliminator,
        the correlations between all features are calculated. If their correlation is higher than a certain threshold,
        the one with the lowest feature importance is eliminated. When using the importance eliminator, features are
        one-by-one eliminated, sorted by their importance. In both situations, the model is then retrained.
        Subsequently, a Kolmogorov-Smirnov analysis is used to find the optimal point where model performance is kept,
        while eliminating as much as possible features.

        :param df: The data.
        :type df: DataFrame
        :param target: Name of the target variable.
        :type target: str
        :param feature_importances: Dict with names of independent variables as keys and with their importance values as values.
        :type feature_importances: dict(str, int)
        :return: None
        :raises TypeError: If one of the input parameters has a wrong type.
        :raises ValueError: If one of the input parameters has a wrong value.
        """

        if not isinstance(self.__cv, int):
            raise TypeError("cv must be of type 'int'.")

        if self.__method == 'correlation':
            if not isinstance(self.__lower_bound, float):
                raise TypeError("If method = 'correlation', lower_bound must be of type 'float'.")

            if not isinstance(self.__step, float):
                raise TypeError("If method = 'correlation', step must be of type 'float'.")
        if self.__method == 'importance':
            if not isinstance(self.__lower_bound, int):
                raise TypeError("If method = 'correlation', lower_bound must be of type 'int'.")

        if not isinstance(self.__method, str):
            raise TypeError("method must be of type 'str'.")

        if not isinstance(self.__p_alpha, float):
            raise TypeError("p_alpha must be of type 'float'.")

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be of type 'pd.DataFrame'.")

        if not isinstance(target, str):
            raise TypeError("target must be of type 'str'.")

        if not isinstance(feature_importances, dict):
            raise TypeError("feature_importances must be of type 'dict', with feature names as keys and importance "
                            "values as values.")

        if (self.__cv < 5) | (self.__cv > 200):
            raise ValueError("cv value must be between [5, 100].")

        if self.__method == 'correlation':
            if (self.__lower_bound < 0) | (self.__lower_bound > .8):
                raise ValueError("lower_bound must be between [0, 0.80].")

            if int(round((1 - self.__lower_bound) / self.__step)) < 8:
                raise ValueError("Too few data can be analyzed, try a smaller step size or a lower lower_bound.")
        else:
            if self.__lower_bound <= 0:
                raise ValueError("lower_bound must be > 0.")

            if len(feature_importances) - self.__lower_bound < 8:
                raise ValueError("Too few data can be analyzed, try a lower lower_bound. If this is already 1, there "
                                 "may be not enough features in your dataset. You need > 9 independent variables.")

        if (self.__step <= 0) | (self.__step > .1):
            raise ValueError("step size must be between (0, 0.1].")

        if self.__method not in ['correlation', 'importance']:
            raise ValueError("Method must be one of ['correlation', 'importance']")

        if (self.__p_alpha < 0.01) | (self.__p_alpha > .25):
            raise ValueError("p_alpha must between [0.01, 0.25].")

        Xdf, y, cols = self._create_initial_data(df, target, feature_importances)

        random_counter = 0
        self.__model_cv_scores = []
        self.__dropped_features = []
        metrics = {'drop_condition': [],
                   'features_left': [],
                   'n_features': [],
                   self.__scoring: []}

        if self.__method == 'correlation':

            self._create_df_models_correlation(df, target, Xdf, y, cols, random_counter, metrics)

        else:  # self.__method == 'importance'

            self._create_df_models_importance(df, target, y, random_counter, metrics)

        self._create_df_ks_analyses()

        self._create_df_ks_analyses_summarized()

        # clean df_ks_analyses
        self.__df_ks_analyses = self.__df_ks_analyses.drop(['batch0_mean', 'batch1_mean'], axis=1)

        self._create_optimal_deterioration_points()

        self._create_features_left()

    def plot(self, plot_type='ks_analysis'):
        """
        Two types of visualizations to explain how the optimal point was found.

        :param plot_type: The type of plot to show. Default is 'ks_analysis'.
        :type plot_type: {'ks_analysis', 'feature_impact'}
        :return: None
        :raises TypeError: If any of the input parameters has the wrong type.
        :raises ValueError: If any of the input parameters has the wrong value.
        """

        if not isinstance(plot_type, str):
            raise TypeError("plot_type must be of type 'str'.")
        if plot_type not in ['ks_analysis', 'feature_impact']:
            raise ValueError("plot_type must be one of ['ks_analysis', 'feature_impact']")

        if plot_type == 'ks_analysis':

            self._plot_ks_analysis()

        else:  # plot_type == 'feature_impact'

            self._plot_feature_impact()
