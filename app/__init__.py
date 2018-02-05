from flask import Flask
from flask_restful import Api
from app.titanic import TitanicSurvival

flask_app = Flask(__name__)
api = Api(flask_app)

api.add_resource(TitanicSurvival, '/classify')


@flask_app.route('/ping')
def ping():
    return 'pong'
