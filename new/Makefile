.PHONY: clean test

test: env
	env/bin/python test.py

env:
	python -m venv --system-site-packages env
	env/bin/pip install -r requirements.txt

fix:
	env/bin/python -m autopep8 -i *.py
	env/bin/python -m docformatter -i *.py


clean:
	rm -rf env || true