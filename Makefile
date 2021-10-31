# Arcane incantation to print all the other targets, from https://stackoverflow.com/a/26339924
help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

# Install exact Python and CUDA versions. Installs rasterio since conda is needed for installation on windows.
conda:
	conda env update --prune -f environment.yml
	echo "RUN THE FOLLOWING COMMAND: conda activate floodwater_mapper"

# (Removed for now since Poetry doesn't work well on Windows) Compile and install exact python packages
poetry:
	pip install poetry
	poetry install

# Lint
lint:
	bash ./tasks/lint.sh

streamlit-app:
	docker build -t streamlit-app:latest .
	docker run -p 8501:8501 streamlit-app:latest