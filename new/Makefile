.PHONY: clean test

test: env
	env/bin/python test.py

env:
	python -m venv --system-site-packages env
	env/bin/pip install -r requirements.txt

clean:
	rm -rf env || true