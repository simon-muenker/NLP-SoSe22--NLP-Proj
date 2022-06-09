#
# Config:

module = classifier
global_cfg = ./experiments/__global.json


# --- --- ---

#
# collected calls:
ex: ex_linguistic ex_transformer ex_hybrid
dev: ex_linguistic_train.0.001 ex_transformer_train.0.001 ex_hybrid_train.0.001

# --- --- ---

#
# linguistic experiment calls
ling_path = ./experiments/linguistic

ex_linguistic: ex_linguistic_train.0.001 ex_linguistic_train.0.010 ex_linguistic_train.0.100 ex_linguistic_train.1.000

ex_linguistic_train.0.001:
	@python3 -m $(module).linguistic -C $(global_cfg) $(ling_path)/__local.json $(ling_path)/train.0.001/__ex.json

ex_linguistic_train.0.010:
	@python3 -m $(module).linguistic -C $(global_cfg) $(ling_path)/__local.json $(ling_path)/train.0.010/__ex.json

ex_linguistic_train.0.100:
	@python3 -m $(module).linguistic -C $(global_cfg) $(ling_path)/__local.json $(ling_path)/train.0.100/__ex.json

ex_linguistic_train.1.000:
	@python3 -m $(module).linguistic -C $(global_cfg) $(ling_path)/__local.json $(ling_path)/train.1.000/__ex.json


# --- --- ---

#
# linguistic experiment calls
trans_path = ./experiments/transformer

ex_transformer: ex_transformer_train.0.001 ex_transformer_train.0.010 ex_transformer_train.0.100 ex_transformer_train.1.000

ex_transformer_train.0.001:
	@python3 -m $(module).transformer -C $(global_cfg) $(trans_path)/__local.json $(trans_path)/train.0.001/__ex.json

ex_transformer_train.0.010:
	@python3 -m $(module).transformer -C $(global_cfg) $(trans_path)/__local.json $(trans_path)/train.0.010/__ex.json

ex_transformer_train.0.100:
	@python3 -m $(module).transformer -C $(global_cfg) $(trans_path)/__local.json $(trans_path)/train.0.100/__ex.json

ex_transformer_train.1.000:
	@python3 -m $(module).transformer -C $(global_cfg) $(trans_path)/__local.json $(trans_path)/train.1.000/__ex.json


# --- --- ---

#
# linguistic experiment calls
hybrid_path = ./experiments/hybrid

ex_hybrid: ex_hybrid_train.0.001 ex_hybrid_train.0.010 ex_hybrid_train.0.100 ex_hybrid_train.1.000

ex_hybrid_train.0.001:
	@python3 -m $(module).hybrid -C $(global_cfg) $(hybrid_path)/__local.json $(hybrid_path)/train.0.001/__ex.json

ex_hybrid_train.0.010:
	@python3 -m $(module).hybrid -C $(global_cfg) $(hybrid_path)/__local.json $(hybrid_path)/train.0.010/__ex.json

ex_hybrid_train.0.100:
	@python3 -m $(module).hybrid -C $(global_cfg) $(hybrid_path)/__local.json $(hybrid_path)/train.0.100/__ex.json

ex_hybrid_train.1.000:
	@python3 -m $(module).hybrid -C $(global_cfg) $(hybrid_path)/__local.json $(hybrid_path)/train.1.000/__ex.json
