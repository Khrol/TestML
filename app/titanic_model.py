from collections import namedtuple
import pandas as pd
import math

from sklearn.externals import joblib

clf = joblib.load('model.pkl')

THRESHOLD = 0.4


class PassengerData(namedtuple('Data', ['sex', 'age', 'sib_sp', 'ticket_class', 'embarked'])):
    def _calc_categorial(self, category, x):
        """
        >>> PassengerData(None, None, 4, None, None)._calc_categorial('sib_sp', 3)
        [0]
        >>> PassengerData(None, None, 3, None, None)._calc_categorial('sib_sp', 3)
        [1]
        >>> PassengerData('male', None, None, None, None)._calc_categorial('sex', 'female')
        [0]
        >>> PassengerData('female', None, None, None, None)._calc_categorial('sex', 'female')
        [1]
        """
        return [int(getattr(self, category) == x)]

    def _calc_known_age(self):
        """
        >>> PassengerData(None, 42, None, None, None)._calc_known_age()
        [1]
        >>> PassengerData(None, None, None, None, None)._calc_known_age()
        [0]
        >>> PassengerData(None, float('nan'), None, None, None)._calc_known_age()
        [0]
        """
        return [int(self.age is not None and not math.isnan(self.age))]

    def _calc_more_10_years(self):
        """
        >>> PassengerData(None, None, None, None, None)._calc_more_10_years()
        [0]
        >>> PassengerData(None, 9, None, None, None)._calc_more_10_years()
        [0]
        >>> PassengerData(None, 10, None, None, None)._calc_more_10_years()
        [1]
        >>> PassengerData(None, 11, None, None, None)._calc_more_10_years()
        [1]
        """
        return [int(self.age >= 10 if self.age else False)]

    def to_features(self):
        """
        >>> PassengerData('female', 5, 4, 1, 'S').to_features()
           sib_sp_4  factorized_sex  sib_sp_3  known_age  more_10_years  class1  \\
        0         1               1         0          1              0       1   
        <BLANKLINE>
           embarkedS  class2  
        0          1       0  
        """
        return pd.DataFrame.from_items((
            ('sib_sp_4', self._calc_categorial('sib_sp', 4)),
            ('factorized_sex', self._calc_categorial('sex', 'female')),
            ('sib_sp_3', self._calc_categorial('sib_sp', 3)),
            ('known_age', self._calc_known_age()),
            ('more_10_years', self._calc_more_10_years()),
            ('class1', self._calc_categorial('ticket_class', 1)),
            ('embarkedS', self._calc_categorial('embarked', 'S')),
            ('class2', self._calc_categorial('ticket_class', 2)),
        ))


def classify(passenger_data):
    score = float(clf.predict(passenger_data.to_features())[0])
    return {'value': score > THRESHOLD, 'score': score}
