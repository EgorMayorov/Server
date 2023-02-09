import numpy as np
import time
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor


def rmse(y1, y2):
    return np.sqrt(((y1 - y2) ** 2).mean())


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        random_seed=153, **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.random_seed = random_seed
        self.__models = []
        self.__importances = []

    def fit(self, X, y, X_val=None, y_val=None, hist=True):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """
        np.random.seed(self.random_seed)
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3
        history = dict.fromkeys(['rmse_train', 'rmse_val', 'time'])
        history['rmse_train'] = []
        history['rmse_val'] = []
        history['time'] = []
        self.__models = []
        self.__importances = []
        allpreds_train = []
        b = False
        if X_val is not None and y_val is not None:
            b = True
            allpreds_val = []

        for i in range(0, self.n_estimators):  # строим деревья
            index = np.random.randint(X.shape[0], size=X.shape[0])
            start_time = time.time()
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         max_features=self.feature_subsample_size,
                                         **self.trees_parameters)
            self.__models.append(tree)
            tree.fit(X[index], y[index])
            self.__importances.append(tree.feature_importances_)
            allpreds_train.append(tree.predict(X))
            pred_train = np.mean(allpreds_train, axis=0)
            history['rmse_train'].append(rmse(y, pred_train))
            if b:
                allpreds_val.append(tree.predict(X_val))
                pred_val = np.mean(allpreds_val, axis=0)
                history['rmse_val'].append(rmse(y_val, pred_val))
            history['time'].append(time.time() - start_time)
        if hist:
            return history

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        pred = np.zeros(X.shape[0])
        for i in range(0, self.n_estimators):
            pred += self.__models[i].predict(X)
        return pred / len(self.__models)

    def get_models(self):
        return self.__models

    def feature_importances_(self):
        return np.mean(self.__importances, axis=0)


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1,
        max_depth=5, feature_subsample_size=None,
        random_seed=153, **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.random_seed = random_seed
        self.__models = []
        self.__w = []
        self.__importances = []

    def fit(self, X, y, X_val=None, y_val=None, hist=True):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        np.random.seed(self.random_seed)
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3

        history = dict.fromkeys(['rmse_train', 'rmse_val', 'time'])
        history['rmse_train'] = []
        history['rmse_val'] = []
        history['time'] = []
        self.__models = []
        self.__w = []
        self.__importances = []
        train_pred = np.zeros(X.shape[0])
        val_pred = 0
        b = False
        if X_val is not None and y_val is not None:
            b = True
            val_pred = np.zeros(X_val.shape[0])
        '''
        idx = np.arange(X.shape[0])
        sample_size = int((1 - 1 / np.e) * X.shape[0])
        '''
        for i in range(0, self.n_estimators):  # строим деревья
            # np.random.shuffle(idx)
            # index = idx[:sample_size]
            index = np.random.randint(X.shape[0], size=X.shape[0])
            start_time = time.time()
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         max_features=self.feature_subsample_size,
                                         **self.trees_parameters)
            self.__models.append(tree)
            s = 2 / len(y) * (y - train_pred)   # остаток
            tree.fit(X[index], s[index])
            self.__importances.append(tree.feature_importances_)
            new_pred = tree.predict(X)
            new_weight = self.learning_rate * minimize_scalar(lambda x:
                                                              ((y - train_pred - x * new_pred) ** 2).mean()).x
            train_pred += new_weight * new_pred
            history['rmse_train'].append(rmse(y, train_pred))
            if b:
                new_val_pred = tree.predict(X_val)
                val_pred += new_weight * new_val_pred
                history['rmse_val'].append(rmse(y_val, val_pred))
            self.__w.append(new_weight)
            history['time'].append(time.time() - start_time)
        if hist:
            return history

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        pred = np.zeros(X.shape[0])
        for i in range(0, self.n_estimators):
            new_pred = self.__models[i].predict(X)
            pred += self.__w[i] * new_pred
        return pred

    def get_models(self):
        return self.__models

    def get_weights(self):
        return self.__w

    def feature_importances_(self):
        return np.mean(self.__importances, axis=0)
