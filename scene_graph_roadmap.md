# Scene Graph Explorer - End-to-End Project Roadmap

> Goal: Build a production-style Computer Vision + Full Stack
> application that converts an uploaded image into an interactive Scene
> Graph.

------------------------------------------------------------------------

# Phase 0 - Project Setup (Day 1)

## Goal

Set up the project and understand the data before writing ML code.

### Tasks

-   [x] Create project repository
-   [x] Create folder structure
-   [x] Download Visual Genome images (VG_100K + VG_100K_2)
-   [x] Download `image_data.json`
-   [x] Add VG150-curated files
    -   VG-SGG.h5
    -   VG-SGG-dicts.json
    -   zeroshot_triplet.pytorch
-   [x] Create Python environment
-   [ ] Install PyTorch, OpenCV, Ultralytics, FastAPI, React
    dependencies (later)

**Deliverable:** Project structure ready.

------------------------------------------------------------------------

# Phase 1 - Dataset Exploration

## Learn

-   Visual Genome structure
-   Scene Graph annotations
-   Objects vs Relationships
-   Bounding boxes
-   Image IDs

### Notebook

`01_dataset_exploration.ipynb`

### Tasks

-   Inspect every key inside VG-SGG.h5
-   Understand VG-SGG-dicts.json
-   Load one image
-   Match image_id with annotations
-   Draw GT bounding boxes
-   Print GT relationships
-   Manually understand one complete scene graph

**Deliverable:** You fully understand the dataset.

------------------------------------------------------------------------

# Phase 2 - Dataset Preprocessing

## Learn

-   Dataset engineering
-   Label mapping
-   Train/Val/Test splits
-   Efficient loading

### Tasks

-   Build Dataset class
-   Build preprocessing pipeline
-   Load image + annotations together
-   Convert annotations into training samples
-   Cache mappings if needed

**Deliverable:** Clean reusable data loader.

------------------------------------------------------------------------

# Phase 3 - Relationship Dataset Creation

## Learn

-   Predicate Classification
-   Subject-Object pairs
-   Positive vs negative pairs

### Tasks

-   Create (subject, object) samples
-   Attach bounding boxes
-   Attach labels
-   Attach predicate
-   Design final training sample format

**Deliverable:** Relationship training dataset.

------------------------------------------------------------------------

# Phase 4 - YOLO-World Integration

## Learn

-   Pretrained object detection
-   Inference pipeline

### Notebook

`02_yoloworld_inference.ipynb`

### Tasks

-   Load YOLO-World
-   Run inference
-   Visualize boxes
-   Compare GT vs predictions
-   Decide confidence threshold

**Deliverable:** Stable detector.

------------------------------------------------------------------------

# Phase 5 - Relationship Model Design

## Learn

-   Feature extraction
-   Predicate Classification
-   Loss functions

### Decide

-   Inputs
-   Feature representation
-   Network architecture
-   Output classes

### Tasks

-   Draw architecture
-   Implement model
-   Unit test forward pass

**Deliverable:** Working PyTorch model.

------------------------------------------------------------------------

# Phase 6 - Training Pipeline

## Learn

-   Dataloaders
-   Training loops
-   Validation
-   Checkpointing

### Tasks

-   Trainer
-   Validation loop
-   Logging
-   Save best model
-   Resume training

**Deliverable:** Trained relationship model.

------------------------------------------------------------------------

# Phase 7 - Evaluation

## Learn

-   Accuracy
-   Recall
-   Confusion matrix
-   Predicate-wise performance

### Tasks

-   Evaluate model
-   Identify weak predicates
-   Save metrics

**Deliverable:** Evaluation report.

------------------------------------------------------------------------

# Phase 8 - Inference Pipeline

## Learn

-   Production inference

Pipeline:

User Image → YOLO-World → Detected Objects → Relationship Model → Scene
Graph

### Tasks

-   Build inference pipeline
-   Convert outputs to graph objects

**Deliverable:** End-to-end ML inference.

------------------------------------------------------------------------

# Phase 9 - Scene Graph Builder

## Learn

-   Property Graphs

### Graph

Node: - label - bbox - confidence

Edge: - predicate - confidence

### Tasks

-   Build graph classes
-   Export graph JSON

**Deliverable:** Complete scene graph.

------------------------------------------------------------------------

# Phase 10 - Backend (FastAPI)

Endpoints:

-   POST /predict
-   GET /graph
-   GET /search
-   GET /image

### Tasks

-   Upload API
-   Run inference
-   Return graph JSON
-   Return overlays

**Deliverable:** Working backend.

------------------------------------------------------------------------

# Phase 11 - Frontend (React)

### Pages

-   Upload
-   Results

### Layout

Left: - Image - Bounding boxes

Right: - React Flow graph

Bottom/Side: - Search - Node details

### Features

-   Click graph → highlight image
-   Click image → highlight graph
-   Search object
-   Search relationships

**Deliverable:** Interactive UI.

------------------------------------------------------------------------

# Phase 12 - Integration

### Tasks

-   Connect React ↔ FastAPI
-   Test end-to-end
-   Improve UX
-   Error handling

**Deliverable:** Functional application.

------------------------------------------------------------------------

# Phase 13 - Deployment

### Backend

-   Docker (optional)
-   Render/Railway/VM

### Frontend

-   Vercel

### Model

-   Store trained weights
-   Load once at startup

**Deliverable:** Live application.

------------------------------------------------------------------------

# Version 1.0 Features

-   Upload image
-   YOLO-World detection
-   Relationship prediction
-   Scene graph generation
-   Graph visualization
-   Object search
-   Relationship search
-   Image ↔ Graph synchronization

------------------------------------------------------------------------

# Version 1.1

-   Attribute prediction
-   Attribute visualization
-   Attribute filtering

------------------------------------------------------------------------

# Version 2.0

-   NLP query parser
-   Natural language graph search

Example: "The person wearing sunglasses drinking coffee"

Parser

↓

Graph query

↓

Highlight matching object

------------------------------------------------------------------------

# Folder Structure

    scene-graph-explorer/

    backend/
    frontend/
    training/
    datasets/
    models/
    checkpoints/
    outputs/
    notebooks/
    docs/

    README.md
    requirements.txt

------------------------------------------------------------------------

# Rule for Every Module

1.  Understand the concept.
2.  Design the solution.
3.  Implement.
4.  Test.
5.  Refactor.
6.  Document.

Never skip understanding for coding.
