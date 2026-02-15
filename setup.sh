git clone https://github.com/krzysztof-kacprzyk/TIMEVIEW/
cp -R TIMEVIEW/experiments/data .

uv venv
uv pip install -e ".[dev]"
