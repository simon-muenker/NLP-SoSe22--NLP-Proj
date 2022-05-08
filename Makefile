module = classifier

dev_transformer:
	@python3 -m $(module).transformer -C ./config/transformer.json


dev_linguistic:
	@python3 -m $(module).linguistic -C ./config/linguistic.json


dev_hybrid:
	@python3 -m $(module).hybrid -C ./config/hybrid.json



exp_transformer:
	@python3 -m $(module).transformer -C ./experiment/base_transformer/config.json