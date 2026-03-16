# Lab 5 — Docker Containerisation

**Prerequisites:** Lab 4 complete. Docker Desktop installed and running.

**What you'll build:** A Docker image that packages your FastAPI app and trained model. You'll run the container locally, then push the image to Docker Hub.

**New concepts:** Dockerfile, image layers, container runtime, Docker Hub, image tagging.

**Time:** ~1 hour

---

## Background

Your API works on your machine. Docker makes it work *identically* on any machine — a colleague's laptop, a cloud VM, a Kubernetes cluster — by packaging your code, dependencies, and model together into a single portable image.

---

## Step 1 — Install Docker Desktop

Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop) and start it. Verify:

```bash
docker --version
```

---

## Step 2 — Create the Dockerfile

Create `Dockerfile` in the project root:

```dockerfile
# Use a slim Python base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy and install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model
COPY src/ ./src/
COPY models/ ./models/

# Expose the port FastAPI will listen on
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Step 3 — Create a .dockerignore

Create `.dockerignore` to keep the image lean:

```
.git
.venv
__pycache__
*.pyc
notebooks/
tests/
data/
mlruns/
labs/
*.md
.dvc
```

---

## Step 4 — Train the model

The model must exist before building the image (it gets copied in at build time):

```bash
python -m src.train
```

---

## Step 5 — Build the image

```bash
docker build -t obesity-classifier:v1 .
```

Watch the layers build. The `pip install` layer will be cached on subsequent builds unless `requirements.txt` changes.

Verify the image exists:

```bash
docker images obesity-classifier
```

---

## Step 6 — Run the container

```bash
docker run -p 8000:8000 obesity-classifier:v1
```

Open `http://localhost:8000/docs` — your API is now running inside the container. Test a prediction exactly as you did in Lab 4.

Press `Ctrl+C` to stop the container.

---

## Step 7 — Push to Docker Hub

1. Create a free account at [hub.docker.com](https://hub.docker.com)
2. Create a new public repository called `obesity-classifier`
3. Log in from the terminal:

```bash
docker login
```

Tag and push:

```bash
# Tag with your Docker Hub username
docker tag obesity-classifier:v1 YOUR_DOCKERHUB_USERNAME/obesity-classifier:v1
docker tag obesity-classifier:v1 YOUR_DOCKERHUB_USERNAME/obesity-classifier:latest

# Push both tags
docker push YOUR_DOCKERHUB_USERNAME/obesity-classifier:v1
docker push YOUR_DOCKERHUB_USERNAME/obesity-classifier:latest
```

4. Visit `hub.docker.com/r/YOUR_DOCKERHUB_USERNAME/obesity-classifier` — your image is publicly available. Anyone can now run it with:

```bash
docker run -p 8000:8000 YOUR_DOCKERHUB_USERNAME/obesity-classifier:latest
```

---

## Step 8 — Commit the Dockerfile

```bash
git add Dockerfile .dockerignore
git commit -m "feat: add Dockerfile for FastAPI app"
git push
```

---

## Key takeaways

- A `Dockerfile` is a recipe: each `RUN`, `COPY`, `FROM` line creates a cached layer
- Order matters: put `COPY requirements.txt` + `RUN pip install` *before* `COPY src/` so that changing your code doesn't invalidate the dependency layer
- `.dockerignore` keeps large/irrelevant files out of the build context
- Docker Hub is the public registry — equivalent to GitHub but for container images
- In Lab 6 you'll automate building and pushing this image via GitHub Actions

---

**Next:** [Lab 6 — Advanced CI/CD Pipeline](lab-06-advanced-cicd.md)
