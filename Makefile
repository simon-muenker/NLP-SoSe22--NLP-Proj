
# force make targets
.PHONY: install download data analysis training

#
# install
install:
	@python3 -m pip install -r requirements.txt

download:
	@wget -O data/_aclImdb.tar.gz http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
	@tar -zxf data/_aclImdb.tar.gz -C data/
	@rm data/_aclImdb.tar.gz
	@mv data/aclImdb data/_aclImdb


# --- --- ---

data:
	@jupyter nbconvert --to notebook --inplace --execute notebooks/00--Data-Loading.ipynb
	@jupyter nbconvert --to notebook --inplace --execute notebooks/01--Data-Preperation.ipynb

analysis:
	@jupyter nbconvert --to notebook --inplace --execute notebooks/02--Manifold-Computation.ipynb
	@jupyter nbconvert --to notebook --inplace --execute notebooks/03--Manifold-Analysis.ipynb

training:
	@jupyter nbconvert --to notebook --inplace --execute notebooks/04--Classifier-Training.ipynb