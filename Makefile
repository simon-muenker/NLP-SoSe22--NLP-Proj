module = classifier

exp_train.1.000:
	@python3 -m $(module).transformer -C ./experiments/train.1.000/linguistic/config.json
	@python3 -m $(module).transformer -C ./experiments/train.1.000/transformer/config.json
	@python3 -m $(module).transformer -C ./experiments/train.1.000/hybrid/config.json