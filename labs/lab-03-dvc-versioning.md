# Lab 3 — Data & Model Versioning with DVC

**Prerequisites:** Lab 2 complete.

**What you'll build:** DVC tracking for your dataset and trained model, plus a local DVC remote so data is stored separately from git.

**New concepts:** Data versioning, DVC remote, `.dvc` pointer files, reproducibility.

**Time:** ~1 hour

---

## Background

Git is great for code but not for large files like datasets and model binaries. DVC (Data Version Control) works alongside git: git tracks a small `.dvc` pointer file, DVC stores the actual data in a separate remote. This means your repo stays lean, but you can always reproduce any historical version.

---

## Step 1 — Install DVC

```bash
pip install dvc
```

Add to `requirements.txt`:
```
dvc>=3.0
```

---

## Step 2 — Initialise DVC

From the root of your project:

```bash
dvc init
git add .dvc .dvcignore
git commit -m "chore: initialise DVC"
```

This creates a `.dvc/` config folder (tracked by git) and a `.dvcignore` file.

---

## Step 3 — Track your dataset with DVC

```bash
dvc add data/ObesityDataSet_Original.csv
```

DVC creates `data/ObesityDataSet_Original.csv.dvc` (a small pointer file) and adds the actual CSV to `.gitignore` automatically.

Add the pointer file to git:

```bash
git add data/ObesityDataSet_Original.csv.dvc data/.gitignore
git commit -m "data: add obesity dataset under DVC tracking"
```

---

## Step 4 — Track your trained model

First train the model if you haven't already:

```bash
python -m src.train
```

Then track the output:

```bash
dvc add models/model.pkl
git add models/model.pkl.dvc models/.gitignore
git commit -m "model: add trained model under DVC tracking"
```

---

## Step 5 — Set up a local DVC remote

A "remote" is where DVC actually stores the data files. For now use a folder on your local machine (in a later lab this can be swapped for cloud storage):

```bash
# Create the remote storage location
mkdir -p ~/dvc-remote

# Register it as the default remote
dvc remote add -d local_remote ~/dvc-remote
git add .dvc/config
git commit -m "chore: configure local DVC remote"
```

Push your data to the remote:

```bash
dvc push
```

You'll see the data and model files copied into `~/dvc-remote`.

---

## Step 6 — Simulate a new data version

Open `data/ObesityDataSet_Original.csv` in a text editor and add a comment line or make a trivial change. Then:

```bash
dvc add data/ObesityDataSet_Original.csv
git add data/ObesityDataSet_Original.csv.dvc
git commit -m "data: v2 — minor dataset update"
dvc push
```

DVC creates a new content-addressed version. The old version is still available — you can get it back with `git checkout` of the old `.dvc` pointer + `dvc pull`.

---

## Step 7 — Verify reproducibility

Delete your local copy of the data and pull it back:

```bash
# Remove the local data file
rm data/ObesityDataSet_Original.csv

# Pull from the DVC remote
dvc pull
```

The file is restored exactly as it was.

---

## Key takeaways

- DVC adds `.dvc` pointer files to git; the actual data lives in the remote
- `dvc add` starts tracking a file; `dvc push` uploads it; `dvc pull` restores it
- Combined with git, you get full reproducibility: every commit maps to a specific version of the data AND the code
- The remote can be a local folder, S3, GCS, Azure Blob — the workflow is identical

---

**Next:** [Lab 4 — REST API Model Serving with FastAPI](lab-04-fastapi-model-serving.md)
