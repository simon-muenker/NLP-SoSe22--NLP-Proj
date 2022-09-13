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

# --- --- ---

analysis:
	@jupyter nbconvert --to notebook --inplace --execute notebooks/02--Manifold-Computation.ipynb
	@jupyter nbconvert --to notebook --inplace --execute notebooks/03--Manifold-Analysis.ipynb