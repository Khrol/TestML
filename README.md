# Envionrment

```bash
$ pipenv shell
```

# Run notebook

```bash
$ jupyter notebook
```

[Presentation Engine](https://damianavila.github.io/RISE/)

# Sample request

```bash
$ curl -X POST -H "Content-Type: application/json" -d \
 '{"sex": "male", "sib_sp": 5, "ticket_class": 2, "embarked": "S"}' \
  http://localhost:5000/classify
```
