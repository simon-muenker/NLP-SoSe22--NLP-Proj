#
# Config:

module = classifier
global_cfg = ./experiments/__global.json

base_path = ./experiments/base
feat_path = ./experiments/features
hybrid_path = ./experiments/hybrid


# --- --- ---

#
# install
install:
	@python3 -m pip install -r requirements.txt

# --- --- ---

#
# collected call:
ex: ex_base ex_features ex_hybrid

ex_base:
	@python3 -m $(module).base -C $(global_cfg) $(base_path)/__ex.json

ex_features:
	@python3 -m $(module).features -C $(global_cfg) $(feat_path)/__ex.json

ex_hybrid:
	@python3 -m $(module).hybrid -C $(global_cfg) $(hybrid_path)/__ex.json


# --- --- ---

#
# debug calls
debug_path = ./experiments/_debug
debug:  debug_base debug_features debug_hybrid

debug_base:
	@python3 -m $(module).base -C $(global_cfg) $(debug_path)/__local.json $(debug_path)/base/__ex.json

debug_features:
	@python3 -m $(module).features -C $(global_cfg) $(debug_path)/__local.json $(debug_path)/features/__ex.json

debug_hybrid:
	@python3 -m $(module).hybrid -C $(global_cfg) $(debug_path)/__local.json $(debug_path)/hybrid/__ex.json
