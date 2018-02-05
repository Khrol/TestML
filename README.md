# Envionrment

```bash
$ pipenv shell
```

# Run notebook

```bash
$ jupyter notebook
```

# Sample request

```bash
$ curl -X POST -H "Content-Type: application/json" -d \
 '{"sex": "male", "sib_sp": 5, "ticket_class": 2, "embarked": "S"}' \
  http://localhost:5000/classify
```
