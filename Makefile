flake8:
	@if command -v flake8 > /dev/null; then \
	        echo "Running flake8"; \
	        flake8 flake8 --ignore N802,N806,F401 `find . -name \*.py | grep -v setup.py | grep -v /docs/ | grep -v /.venv/ | grep -v /sphinx/`; \
	else \
	        echo "flake8 not found, please install it!"; \
	        exit 1; \
	fi;
	@echo "flake8 passed"

test:
	python -m pytest

conda_env := static-site-generation
notebooks := $(wildcard notebooks/*.ipynb) $(wildcard notebooks/**/*.ipynb)
md_pages := $(patsubst notebooks/%.ipynb,docs/%.md,$(notebooks))

build.env: ; conda env create -f environment.yml
build.site: $(md_pages)

clean.env: ; conda remove --name $(conda_env) --all
clean.site: ; rm $(md_pages)

print-%: ; @echo $* is $($*):

docs/%.md: notebooks/%.ipynb
	jupyter nbconvert\
		--to markdown $<\
		--output-dir $(dir $@)