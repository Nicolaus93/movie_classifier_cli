from movie_classifier.cli import CLInterface
from pathlib import Path


p = Path('./movie_classifier') / "data"
new_cli = CLInterface(p.absolute())
new_cli.parse()
res = new_cli.predict()
print(res)
