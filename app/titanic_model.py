from collections import namedtuple
import pandas as pd

from sklearn.externals import joblib

clf = joblib.load('model.pkl')

THRESHOLD = 0.4


class PassengerData(namedtuple('Data', ['sex', 'age', 'sib_sp', 'ticket_class', 'embarked'])):
    def to_features(self):
        return pd.DataFrame.from_items((
            ('sib_sp_4', [int(self.sib_sp == 4)]),
            ('factorized_sex', [int(self.sex == 'male')]),
            ('sib_sp_3', [int(self.sib_sp == 3)]),
            ('known_age', [int(self.age is None)]),
            ('more_10_years', [int(self.age > 10 if self.age else False)]),
            ('sib_sp_5', [int(self.sib_sp == 5)]),
            ('embarkedS', [int(self.embarked == 'S')]),
            ('class3', [int(self.ticket_class == 3)]),
            ('class1', [int(self.ticket_class == 1)])
        ))


def classify(passenger_data):
    input_data = passenger_data.to_features()
    score = float(clf.predict(passenger_data.to_features())[0])
    print(input_data, score)
    return {'value': score > THRESHOLD, 'score': score}
