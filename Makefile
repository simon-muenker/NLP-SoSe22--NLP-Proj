module = classifier

ex_train.1.000_linguistic:
	@python3 -m $(module).linguistic -C ./experiments/train.1.000/linguistic/config.json

ex_train.1.000_transformer:
	@python3 -m $(module).transformer -C ./experiments/train.1.000/transformer/config.json

ex_train.1.000_hybrid:
	@python3 -m $(module).hybrid -C ./experiments/train.1.000/hybrid/config.json

exp_train.1.000: ex_train.1.000_linguistic ex_train.1.000_transformer ex_train.1.000_hybrid