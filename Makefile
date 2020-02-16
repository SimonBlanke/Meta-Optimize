dist:
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install:
	pip install .

develop:
	pip install -e .

reinstall:
	pip uninstall -y meta_optimize
	rm -fr build dist meta_optimize.egg-info
	python setup.py bdist_wheel
	pip install dist/*
