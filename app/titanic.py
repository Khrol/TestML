from flask_restful import Resource, reqparse
from app.titanic_model import PassengerData, classify


class TitanicSurvival(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('sex', type=str, required=True)
        self.parser.add_argument('age', type=float, required=False)
        self.parser.add_argument('sib_sp', type=int, required=True)
        self.parser.add_argument('ticket_class', type=int, required=True)
        self.parser.add_argument('parch', type=int, required=True)
        self.parser.add_argument('embarked', type=str, required=True)
        super(TitanicSurvival, self).__init__()

    def post(self):
        args = self.parser.parse_args()
        return classify(PassengerData(**args))
