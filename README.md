# Envionrment

```bash
$ conda env create -f environment.yml
$ conda activate test_ml
```

# Run notebook

```bash
$ jupyter notebook
```

[Presentation Engine](https://damianavila.github.io/RISE/)

# Sample request

Run application
```bash
$ python ./main.py
```

```bash
$ curl -X POST -H "Content-Type: application/json" -d \
 '{"sex": "male", "sib_sp": 5, "ticket_class": 2, "embarked": "S", "parch": 2}' \
  http://localhost:5000/classify
```

# Run tests

```bash
$ python -m pytest --doctest-modules
```