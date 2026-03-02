Implementation is based on https://github.com/karpathy/ng-video-lecture
with customization and improvements

## Environment setup (uv + Python 3.12)

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv sync --dev
```

## Tooling

```bash
uv run ruff check .
uv run ruff format .
uv run pre-commit run --all-files
```
