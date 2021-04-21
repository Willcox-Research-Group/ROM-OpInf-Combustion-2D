.PHONY: lint install


PYTHON = python3
LINTER = flake8
INSTALLER = pip install --user -r


lint:
	$(PYTHON) -m $(LINTER) *.py


install:
	$(PYTHON) -m $(INSTALLER) requirements.txt

