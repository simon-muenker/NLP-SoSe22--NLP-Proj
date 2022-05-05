module = classifier

transformer:
	@python3 -m $(module).transformer -C ./config/transformer.json