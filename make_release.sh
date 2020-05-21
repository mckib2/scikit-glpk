# Remove any existing distribution archives
rm -rf dist
mkdir dist

# Generate distribution archives
python -m pip install --upgrade setuptools wheel
python setup.py sdist bdist_wheel

# Upload
python -m pip install --upgrade twine
python -m twine upload dist/*
