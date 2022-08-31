#
# Config:

module = classifier
global_cfg = ./experiments/__global.json

# --- --- ---

#
# install
install:
	@python3 -m pip install -r requirements.txt

# --- --- ---

experiment:
	@python3 -m $(module).base -C $(global_cfg) $(base_path)/__ex.json


# --- --- ---

#
# debug call
debug:
	@python3 -m $(module) -C $(global_cfg) ./experiments/_debug/config.json

