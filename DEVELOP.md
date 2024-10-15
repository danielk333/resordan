

### linting with flake8

To lint a specific file using this projects setup, run

```bash
flake8 --config setup.cfg ./src/resordan/myfile.py
```

or on a entire folder

```bash
flake8 --config setup.cfg ./src/resordan/myfolder/
```

#### flake8 linting with Code (VS / OSS)

Open command palette and set the python interpreter to your local environment 
by searching for `Python: select interpreter`. 

Then select flake8 as the linter by searching for `Python: select linter` in 
the command palette. 

### formatting with black

To format a single file, run

```bash
black --config pyproject.toml ./src/hardtarget/file.py
```

or target an entire folder with 

```bash
black --config pyproject.toml ./src/hardtarget/tools/
```

#### To enable auto-formatting in Code (VS / OSS)

Add the black provider to Code's config with

```json
"python.formatting.provider": "black",
```

Then auto-formatting of the current file is by default bound to `Ctrl+Shift+i` 
and can be changed by searching for the keybinding `Format document`.

### pytest

To run entire suite
```bash
pytest
```
To run specific file
```bash
pytest tests/test_myfile.py
```

To run specific test function, within file
```bash
pytest tests/test_myfile.py::my_function


To run specific TestCase, within file
```bash
pytest tests/test_myfile.py::MyTest
```

To run specific test function within TestCase
```bash
pytest tests/test_myfile.py::MyTest::test_myfunction
```

To run interactive tests
```bash
pytest -m interactive tests/test_myfile.py
```

