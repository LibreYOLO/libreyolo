# Documentation

LibreYOLO uses **Sphinx** with the **Furo** theme for documentation.

## Quick Start

```bash
# Install dependencies
pip install -r docs_sphinx/requirements.txt

# Build docs
cd docs_sphinx
make html

# View locally
open _build/html/index.html
# or serve with live preview
python -m http.server -d _build/html 8000
```

## How It Works

The documentation is generated from **two sources**:

1. **Manual pages** (markdown files in `docs_sphinx/`)
   - `index.md` — Homepage
   - `getting-started.md` — Installation guide
   - `inference.md` — Inference options
   - `training.md` — Training guide  
   - `explainability.md` — CAM methods

2. **Auto-generated API docs** (from Python docstrings)
   - `api/index.md` contains `autoclass::` directives
   - Sphinx reads your docstrings and generates documentation

Example in `api/index.md`:
```rst
.. autoclass:: libreyolo.LIBREYOLO8
   :members:
   :show-inheritance:
```

This reads the docstring from `LIBREYOLO8` and renders it as documentation.

## Deployment Options

### GitHub Pages (Recommended)

A GitHub Actions workflow is ready at `.github/workflows/docs.yml`.

**To enable:**
1. Go to repo Settings → Pages
2. Set Source to "GitHub Actions"
3. Push to `main` branch

Docs will auto-deploy on every push.

### Manual Deploy

```bash
cd docs_sphinx
make html
# Push _build/html to gh-pages branch
```

### ReadTheDocs (Free for Open Source)

1. Go to [readthedocs.org](https://readthedocs.org)
2. Import your GitHub repo
3. It auto-detects `.readthedocs.yaml`
4. Docs build automatically on every push

### Vercel

1. Connect your repo to Vercel
2. Build Command: `cd docs_sphinx && make html`
3. Output Directory: `docs_sphinx/_build/html`

## File Structure

```
docs_sphinx/
├── conf.py              # Sphinx configuration
├── index.md             # Homepage
├── getting-started.md   # Installation guide
├── inference.md         # Inference guide
├── training.md          # Training guide
├── explainability.md    # XAI guide
├── api/
│   └── index.md         # API reference (autodoc)
├── _static/             # Custom CSS/images
├── requirements.txt     # Sphinx dependencies
├── Makefile             # Build commands
└── .readthedocs.yaml    # ReadTheDocs config
```

## Adding New Documentation

1. Create a new `.md` file in `docs_sphinx/`:
   ```bash
   touch docs_sphinx/new-feature.md
   ```

2. Write your content with markdown

3. Add it to the `toctree` in `index.md`:
   ```markdown
   ```{toctree}
   :maxdepth: 2
   :caption: User Guide

   getting-started
   inference
   training
   explainability
   new-feature        # ← Add here
   ```
   ```

4. Rebuild: `make html`

## Updating API Docs

**No manual updates needed!**

When you update a docstring in your code:
```python
class LIBREYOLO8:
    """
    Updated docstring here.  # ← Change this
    """
```

Just rebuild the docs — the API reference updates automatically.

## Typical Maintenance Workflow

```bash
# 1. Make changes to code docstrings or docs markdown files

# 2. Rebuild locally
cd docs_sphinx && make html

# 3. Preview
open _build/html/index.html

# 4. Commit & push
git add .
git commit -m "docs: update inference guide"
git push

# 5. GitHub Pages auto-deploys ✅
```

## Live Reload (Optional)

For auto-rebuild while editing:

```bash
pip install sphinx-autobuild
sphinx-autobuild docs_sphinx docs_sphinx/_build/html
```

Open http://127.0.0.1:8000 — pages refresh automatically when you save!

## Dependencies

```
sphinx>=7.0
furo>=2024.0
myst-parser>=2.0
```

These are in `docs_sphinx/requirements.txt`.
