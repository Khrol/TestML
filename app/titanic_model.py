from collections import namedtuple
import pandas as pd
import math

from sklearn.externals import joblib

clf = joblib.load('model.pkl')

THRESHOLD = 0.415


class PassengerData(namedtuple('Data', ['sex', 'age', 'sib_sp', 'ticket_class', 'parch', 'embarked'])):
    def _calc_categorial(self, category, x):
        """
        >>> PassengerData(None, None, 4, None, None, None)._calc_categorial('sib_sp', 3)
        [0]
        >>> PassengerData(None, None, 3, None, None, None)._calc_categorial('sib_sp', 3)
        [1]
        >>> PassengerData('male', None, None, None, None, None)._calc_categorial('sex', 'female')
        [0]
        >>> PassengerData('female', None, None, None, None, None)._calc_categorial('sex', 'female')
        [1]
        """
        return [int(getattr(self, category) == x)]

    def _calc_known_age(self):
        """
        >>> PassengerData(None, 42, None, None, None, None)._calc_known_age()
        [1]
        >>> PassengerData(None, None, None, None, None, None)._calc_known_age()
        [0]
        >>> PassengerData(None, float('nan'), None, None, None, None)._calc_known_age()
        [0]
        """
        return [int(self.age is not None and not math.isnan(self.age))]

    def _calc_more_x_years(self, x):
        """
        >>> PassengerData(None, None, None, None, None, None)._calc_more_x_years(10)
        [0]
        >>> PassengerData(None, 9, None, None, None, None)._calc_more_x_years(10)
        [0]
        >>> PassengerData(None, 10, None, None, None, None)._calc_more_x_years(10)
        [1]
        >>> PassengerData(None, 11, None, None, None, None)._calc_more_x_years(10)
        [1]
        """
        return [int(self.age >= x if self.age else False)]

    def to_features(self):
        """
        >>> PassengerData('female', 5, 4, 1, 2, 'S').to_features()
           factorized_sex  class1  sib_sp_4  known_age  sib_sp_3  more_10_years  \\
        0               1       1         1          1         0              0   
        <BLANKLINE>
           class2  embarkedS  more_40_years  parch2  
        0       0          1              0       1  
        """
        return pd.DataFrame.from_items((
            ('factorized_sex', self._calc_categorial('sex', 'female')),
            ('class1', self._calc_categorial('ticket_class', 1)),
            ('sib_sp_4', self._calc_categorial('sib_sp', 4)),
            ('known_age', self._calc_known_age()),
            ('sib_sp_3', self._calc_categorial('sib_sp', 3)),
            ('more_10_years', self._calc_more_x_years(10)),
            ('embarkedS', self._calc_categorial('embarked', 'S')),
            ('class2', self._calc_categorial('ticket_class', 2)),
            ('more_40_years', self._calc_more_x_years(40)),
            ('parch2', self._calc_categorial('parch', 2)),
        ))


def classify(passenger_data):
    input_dataframe = passenger_data.to_features()
    score = float(clf.predict_proba(input_dataframe).T[1][0])
    print(input_dataframe)
    print(score)
    return {'value': score > THRESHOLD, 'score': score}
