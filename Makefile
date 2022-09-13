
#
# install
install:
	@python3 -m pip install -r requirements.txt


# --- --- ---

data:
	@jupyter nbconvert --to notebook --inplace --execute notebooks/00--Data-Loading.ipynb
	@jupyter nbconvert --to notebook --inplace --execute notebooks/01--Data-Preperation.ipynb

analysis:
	@jupyter nbconvert --to notebook --inplace --execute notebooks/02--Manifold-Computation.ipynb
	@jupyter nbconvert --to notebook --inplace --execute notebooks/03--Manifold-Analysis.ipynb

training:
	@jupyter nbconvert --to notebook --inplace --execute notebooks/04--Classifier-Training.ipynb