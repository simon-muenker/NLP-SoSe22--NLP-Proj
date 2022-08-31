#
# Config:

module = classifier
global_cfg = ./experiment/__global.json

# --- --- ---

#
# install
install:
	@python3 -m pip install -r requirements.txt

# --- --- ---

run:
	@python3 -m $(module) -C $(global_cfg) ./experiment/config.json

debug:
	@python3 -m $(module) -C $(global_cfg) ./experiment/_debug/config.json

