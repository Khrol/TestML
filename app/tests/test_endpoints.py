from app import flask_app
import json

EPS = 0.000001


def test_ping():
    flask_app.testing = True
    client = flask_app.test_client()
    r = client.get('/ping')
    assert r.status_code == 200
    assert 'pong' in r.data.decode('utf-8')


def test_classifier():
    flask_app.testing = True
    client = flask_app.test_client()
    r = client.post('/classify',
                    content_type='application/json',
                    data='{"sex": "female", "sib_sp": 3, "ticket_class": 1, "embarked": "S", "parch": 2}')
    assert r.status_code == 200
    data = json.loads(r.data.decode('utf-8'))
    assert data['value']
    assert abs(data['score'] - 0.5633669) < EPS
