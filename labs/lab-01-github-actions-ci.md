# Lab 1 — GitHub Actions: Your First CI Pipeline

**Prerequisites:** Completed the [quickstart](../README.md#quickstart) — `pytest tests/ -v` passes locally.

**What you'll build:** A GitHub Actions workflow that automatically runs linting and tests on every push and pull request.

**New concepts:** CI pipeline, event-driven automation, workflow YAML syntax, status badges.

**Time:** ~45 minutes

---

## Background

Right now your tests only run when *you* remember to run them. A CI (Continuous Integration) pipeline fixes that — every time code is pushed to GitHub, the tests run automatically. If they fail, you know immediately before the broken code affects anyone else.

---

## Step 1 — Create the workflow file

Create the file `.github/workflows/ci.yml` in your repo:

```yaml
name: CI

on:
  push:
    branches: ["**"]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Lint with flake8
        run: flake8 src/ tests/

      - name: Run tests
        run: pytest tests/ -v
```

Commit and push:

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions CI pipeline"
git push
```

---

## Step 2 — Watch it run

1. Go to your repo on GitHub
2. Click the **Actions** tab
3. You'll see your workflow running — click it to watch the live log

When it completes, you'll see a green ✅ next to your commit.

---

## Step 3 — See it fail

Now intentionally break a test to confirm the pipeline catches it:

Open `tests/test_evaluate.py` and change the accuracy threshold:

```python
# Change this line:
assert eval_bundle["accuracy"] > 0.90

# To this (impossibly high threshold):
assert eval_bundle["accuracy"] > 0.999
```

Commit and push:

```bash
git add tests/test_evaluate.py
git commit -m "test: intentionally failing test"
git push
```

Go to the Actions tab — you'll see a red ❌. The pipeline blocked the bad code.

Now revert the change:

```bash
git add tests/test_evaluate.py
git commit -m "test: revert threshold to correct value"
git push
```

Green again. ✅

---

## Step 4 — Add a status badge to your README

On GitHub, go to **Actions → CI → the three dots (...)** → **Create status badge** → copy the markdown.

Paste it at the top of your `README.md`:

```markdown
![CI](https://github.com/YOUR_USERNAME/mlops-obesity-classifier/actions/workflows/ci.yml/badge.svg)
```

---

## What you built

Every push to this repo now automatically:
1. Checks out your code on a fresh Ubuntu machine
2. Installs your dependencies
3. Runs flake8 linting across `src/` and `tests/`
4. Runs all pytest tests

If any step fails, the pipeline turns red and you see exactly which step failed and why.

---

## Key takeaways

- GitHub Actions is triggered by **events** (push, pull_request, schedule, etc.)
- A **workflow** is a YAML file in `.github/workflows/`
- A workflow has one or more **jobs**, each with **steps**
- `uses:` references a pre-built action; `run:` executes a shell command
- The CI pipeline is the first line of defence — it catches problems before they reach production

---

**Next:** [Lab 2 — Experiment Tracking with MLflow](lab-02-mlflow-experiment-tracking.md)
