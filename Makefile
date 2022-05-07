module = classifier

transformer:
	@python3 -m $(module).transformer -C ./config/transformer.json


linguistic:
	@python3 -m $(module).linguistic -C ./config/linguistic.json