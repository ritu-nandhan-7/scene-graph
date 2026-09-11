# Scene Graph Relationship Prediction Project

## Complete Training and Testing Workflow

**Project:** Scene Graph Relationship Prediction\
**Testing notebook:** `05_testing.ipynb`\
**Final checkpoint:** `best_full_fusion_30epoch.pt`

------------------------------------------------------------------------

# 1. Objective

The objective of this component is to predict the semantic relationship
between two objects in an image.

The final intended pipeline is:

``` text
Image
  ↓
Object Detection
  ↓
Objects + Bounding Boxes
  ↓
Subject–Object Pair Generation
  ↓
Visual Features + Object Labels + Geometry
  ↓
Relationship Model
  ↓
Predicate / No Relationship
  ↓
Scene Graph Triplets
```

Examples:

``` text
person → riding → bike
person → holding → phone
sign → on → building
person → no relationship → car
```

The work progressed from cached feature training to Kaggle GPU training,
local checkpoint restoration, ground-truth image testing, YOLO
integration, and end-to-end relationship prediction.

------------------------------------------------------------------------

# 2. Dataset

The project uses the Visual Genome Scene Graph Generation dataset.

Dataset structure:

``` text
datasets/visual_genome/
├── VG-SGG.h5
├── VG-SGG-dicts.json
├── image_data.json
├── VG_100K/
└── VG_100K_2/
```

  File                     Purpose
  ------------------------ -------------------------------------
  `VG-SGG.h5`              Object and relationship annotations
  `VG-SGG-dicts.json`      Object and predicate mappings
  `image_data.json`        Image metadata
  `VG_100K`, `VG_100K_2`   Dataset images

The relationship setup contains:

``` text
150 object classes
50 Visual Genome predicate classes
1 no-relationship class
------------------------------
51 total model output classes
```

Final class convention:

``` text
0      → no relationship
1–50   → actual Visual Genome predicates
```

------------------------------------------------------------------------

# 3. Visual Genome Loader

The repository already contained a custom Visual Genome loader with:

``` text
BoundingBox
ImageInfo
Relationship
Scene
SceneObject
VisualGenomeLoader
draw_scene
show_scene
print_image_info
print_objects
print_relationships
```

The main API is:

``` python
loader = VisualGenomeLoader(dataset_root, image_root)
scene = loader.load_scene(image_idx)
```

A scene contains:

``` text
Scene
├── PIL image
├── image metadata
├── objects
└── relationships
```

Each object includes its class label, Visual Genome label ID and
bounding box.

Each relationship includes:

``` text
subject
object
predicate
predicate_index
```

------------------------------------------------------------------------

# 4. Object Label Indexing

Visual Genome object IDs are 1-based:

``` text
person    → 95
shirt     → 114
bike      → 14
car       → 30
vehicle   → 142
shoe      → 115
tree      → 138
```

The PyTorch embeddings use zero-based indices, so inference must use:

``` text
VG ID - 1 = embedding ID
```

Examples:

``` text
person    : 95 → 94
shirt     : 114 → 113
bike      : 14 → 13
car       : 30 → 29
vehicle   : 142 → 141
shoe      : 115 → 114
tree      : 138 → 137
```

This conversion had to match the training pipeline exactly.

------------------------------------------------------------------------

# 5. Cached Feature Training Strategy

Running the visual encoder repeatedly for every object pair and every
epoch would be expensive.

Therefore, visual features were precomputed and stored in cache shards.

Each sample contains:

``` text
visual
subject_label
object_label
geometry
target
image_index
```

A validation shard contained:

``` text
visual             shape=(10000, 512)
subject_label      shape=(10000,)
object_label       shape=(10000,)
geometry           shape=(10000, 8)
target             shape=(10000,)
image_index        shape=(10000,)
```

Local cache:

``` text
notebook/models/preprocessed_cache/
├── train/
│   ├── train_0000.pt ... train_0075.pt
│   ├── train_image_indices.pt
│   └── train_relationship_samples.pt
└── val/
    └── val_0000.pt ... val_0008.pt
```

The model therefore trains directly from cached 512-dimensional visual
features.

------------------------------------------------------------------------

# 6. Cached Dataset

The dataset implementation used:

``` python
class CachedRelationshipDataset(Dataset):

    def __init__(self, cache):
        self.visual = cache["visual"]
        self.subject_label = cache["subject_label"]
        self.object_label = cache["object_label"]
        self.geometry = cache["geometry"]
        self.target = cache["target"]
        self.image_index = cache["image_index"]

    def __getitem__(self, index):
        return {
            "visual": self.visual[index],
            "subject_label": self.subject_label[index] - 1,
            "object_label": self.object_label[index] - 1,
            "geometry": self.geometry[index],
            "target": self.target[index],
            "image_index": self.image_index[index],
        }
```

Important details:

-   Object labels in the cache are 1-based.
-   They are converted to 0-based embedding indices.
-   Targets already represent output classes `0–50`.

------------------------------------------------------------------------

# 7. Geometry Representation

Each pair contains:

``` text
[
 subject_cx,
 subject_cy,
 subject_width,
 subject_height,
 object_cx,
 object_cy,
 object_width,
 object_height
]
```

Thus:

``` text
Raw geometry dimension = 8
```

The model learns a 32-dimensional representation from this spatial
input.

------------------------------------------------------------------------

# 8. CachedRelationshipModel

The model used for Kaggle training was initially defined directly inside
the Kaggle notebook, so it did not exist locally.

It was recreated in `models/relationship/model.py`.

Architecture:

``` python
class CachedRelationshipModel(nn.Module):

    def __init__(
        self,
        num_object_classes=150,
        num_predicate_classes=51,
        visual_dim=512,
        label_embedding_dim=64,
        geometry_dim=32,
        hidden_dim=256,
        dropout=0.30,
    ):
        super().__init__()

        self.subject_embedding = nn.Embedding(150, 64)
        self.object_embedding = nn.Embedding(150, 64)

        self.geometry_encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(672, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 51),
        )
```

The forward pass:

``` text
visual feature
+
subject embedding
+
object embedding
+
geometry encoding
↓
feature concatenation
↓
relationship classifier
↓
51 logits
```

------------------------------------------------------------------------

# 9. Feature Fusion

The four information sources are:

  Source                Dimension
  ------------------- -----------
  Visual feature              512
  Subject embedding            64
  Object embedding             64
  Encoded geometry             32

Total:

``` text
512 + 64 + 64 + 32 = 672
```

Classifier:

``` text
672
 ↓
Linear
 ↓
256
 ↓
ReLU
 ↓
Dropout
 ↓
Linear
 ↓
51 classes
```

------------------------------------------------------------------------

# 10. Kaggle Training

The cached relationship dataset was uploaded to Kaggle as:

``` text
relationship-training-backup
```

Dataset reference:

``` text
kaggle.com/datasets/ritunandhan/relationship-training-backup/versions/2
```

Kaggle was used for GPU training because the full dataset contained
hundreds of thousands of samples.

The original configuration was:

``` text
alpha                    : 0.35
learning_rate            : 0.0005
weight_decay             : 0.0001
epochs                   : 30
batch_size               : 128
dropout                  : 0.3
architecture             : CachedRelationshipModel
inputs                   : visual + subject label + object label + geometry
```

Optimizer:

``` python
optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
```

Loss:

``` python
criterion = nn.CrossEntropyLoss()
```

------------------------------------------------------------------------

# 11. Original 30-Epoch Training Result

Training completed with:

``` text
30-EPOCH FULL-FUSION TRAINING COMPLETE

Best epoch        : 7
Best Val Macro F1 : 0.2759
Checkpoint        : /kaggle/working/best_full_fusion_15epoch.pt
```

The checkpoint filename still said `15epoch`, but the actual run was
configured for 30 epochs.

Therefore:

``` text
Training duration = 30 epochs
Best checkpoint = epoch 7
```

The checkpoint was downloaded locally as:

``` text
best_full_fusion_30epoch.pt
```

------------------------------------------------------------------------

# 12. Checkpoint Contents

The saved dictionary contained:

``` text
epoch
model_state_dict
optimizer_state_dict
val_macro_f1
val_accuracy
val_loss
val_top5_accuracy
history
config
```

Saved metrics:

``` text
Best epoch       : 7
Val Macro F1     : 0.275937
Val Accuracy     : 0.680220
Val Loss         : 1.419596
Val Top-5        : 0.957363
```

The state dictionary verified the architecture:

``` text
subject_embedding.weight      (150, 64)
object_embedding.weight       (150, 64)
geometry_encoder.0.weight     (32, 8)
geometry_encoder.0.bias       (32,)
classifier.0.weight           (256, 672)
classifier.0.bias             (256,)
classifier.3.weight           (51, 256)
classifier.3.bias             (51,)
```

This confirmed that the locally recreated architecture matched the
checkpoint.

------------------------------------------------------------------------

# 13. Local Testing and Import Fix

Testing was performed locally in:

``` text
05_testing.ipynb
```

on:

``` text
Device: CPU
```

Initially, importing the model caused:

``` text
ModuleNotFoundError: No module named 'datasets'
```

The reason was `models/relationship/__init__.py`.

Python executed the package initializer, which imported:

``` python
from .dataset import RelationshipDataset
```

That module depended on the Visual Genome package path.

The initializer was simplified to:

``` python
from .model import RelationshipModel, CachedRelationshipModel

__all__ = [
    "RelationshipModel",
    "CachedRelationshipModel",
]
```

After restarting the kernel, imports worked.

------------------------------------------------------------------------

# 14. Loading the Checkpoint

The checkpoint was loaded using:

``` python
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)
```

The exact model was recreated:

``` python
model = CachedRelationshipModel(
    num_object_classes=150,
    num_predicate_classes=51,
    visual_dim=512,
    label_embedding_dim=64,
    geometry_dim=32,
    hidden_dim=256,
    dropout=0.30,
).to(device)
```

Weights:

``` python
model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()
```

------------------------------------------------------------------------

# 15. First Cached-Sample Inference

A single cached validation sample was tested:

``` text
Visual       : (1, 512)
Subject      : 65
Object       : 134
Geometry     : (1, 8)
Target       : 27
Prediction   : 19
Logits       : (1, 51)
```

This verified:

-   correct checkpoint loading
-   correct architecture
-   correct input dimensions
-   successful forward pass
-   valid 51-class output

The sample was misclassified, but the infrastructure worked.

------------------------------------------------------------------------

# 16. Ground-Truth Image Testing

The next goal was testing with an actual Visual Genome image.

Before using YOLO, testing deliberately used:

``` text
Ground-truth object labels
Ground-truth bounding boxes
Ground-truth relationships
```

This isolates relationship-model performance from detector errors.

Pipeline:

``` text
Visual Genome image
 ↓
Choose subject and object
 ↓
Create union bounding box
 ↓
Crop union region
 ↓
Preprocess to 224 × 224
 ↓
Visual encoder
 ↓
512-D feature
 ↓
Prepare labels and geometry
 ↓
Relationship model
```

------------------------------------------------------------------------

# 17. Union Region and Visual Feature

For a pair such as:

``` text
person + shirt
```

a union crop covering both objects is generated.

The image preprocessing output was:

``` text
Shape: (1, 3, 224, 224)
```

The visual encoder generated:

``` text
Shape: (1, 512)
Dtype: torch.float32
```

This matches the cached training feature dimension.

------------------------------------------------------------------------

# 18. Ground-Truth Relationship Example

One real image test:

``` text
Subject      : person
Object       : shirt
Ground truth : has
Prediction   : watching
Confidence   : 72.66%
```

Result:

``` text
❌ Incorrect
```

Even though the semantic prediction was wrong, the complete
image-to-model pipeline worked correctly.

------------------------------------------------------------------------

# 19. Exact Match Example

Another test:

``` text
Subject      : sign
Object       : building
Ground truth : on
Prediction   : on
Confidence   : 58.64%
```

Result:

``` text
✅ EXACT MATCH
```

This proved that the model could correctly predict some real image
relationships.

------------------------------------------------------------------------

# 20. Stage 2: YOLO Integration

The next stage used detected objects instead of ground truth.

Pipeline:

``` text
Input Image
 ↓
YOLO Detection
 ↓
Object Labels + Boxes + Confidence
 ↓
Map to Visual Genome classes
 ↓
Generate compatible ordered pairs
 ↓
Union crop
 ↓
Visual feature extraction
 ↓
Relationship prediction
```

An example detection set included:

``` text
person
person
car
bike
vehicle
shoe
sidewalk
tree
tree
```

These labels were converted into relationship-model-compatible IDs.

------------------------------------------------------------------------

# 21. Ordered Pair Generation

For `N` objects:

``` text
N × (N - 1)
```

ordered pairs are possible.

For nine objects:

``` text
9 × 8 = 72 possible ordered pairs
```

A compatibility filter reduced these to meaningful pairs.

Example:

``` text
person → car
person → bike
car → person
car → bike
bike → person
bike → car
```

The tested image had:

``` text
10 valid pairs
```

------------------------------------------------------------------------

# 22. Predicate Mapping Bug

Inference initially raised:

``` text
KeyError: 50
```

The Visual Genome dictionary contained 50 predicate IDs:

``` text
above      → 1
across     → 2
...
worn by    → 50
```

The model had 51 output classes:

``` text
0–50
```

The missing class was:

``` text
0 → no relationship
```

Correct mapping:

``` text
0  → no relationship
1  → above
2  → across
...
50 → worn by
```

This was a critical inference-side correction.

------------------------------------------------------------------------

# 23. Why `worn by` Appeared Everywhere

Before fixing and investigating the prediction behaviour, outputs
included:

``` text
person → worn by → car
person → worn by → bike
car → worn by → person
car → worn by → bike
bike → worn by → person
bike → worn by → car
```

One hypothesis was that the model was predicting invalid class numbers
and somehow capping them at 50.

This was ruled out.

The model output is:

``` python
argmax(logits)
```

and therefore can only produce one of its valid 51 output indices.

The model was genuinely predicting class 50 very frequently.

------------------------------------------------------------------------

# 24. Visual Feature Distribution Check

YOLO-derived feature:

``` text
Shape : (1, 512)
Mean  : 0.01474
Std   : 0.50381
Min   : -8.61658
Max   : 1.41959
Norm  : 11.39369
```

Cached training features:

``` text
Shape : (10000, 512)
Mean  : -0.00677
Std   : 0.49857
Min   : -11.59765
Max   : 3.56659
Sample norm : 11.28448
```

The feature distributions were reasonably similar.

Therefore, feature incompatibility was not the main explanation for the
class-50 behaviour.

------------------------------------------------------------------------

# 25. Cached Validation Sanity Check

The checkpoint was evaluated directly on cached validation samples.

Result for 1000 samples:

``` text
Correct  : 660
Accuracy : 0.6600
```

This was close to the saved validation accuracy.

Therefore:

``` text
Checkpoint loading was correct.
Local architecture matched.
Cached inference worked.
```

However, prediction distribution revealed a major bias.

------------------------------------------------------------------------

# 26. Prediction Distribution

Among 1000 validation predictions:

``` text
Class 50 predictions : 479
Class 0 predictions  : 5
```

Class 50 is:

``` text
worn by
```

This pointed directly toward a training-data imbalance problem.

------------------------------------------------------------------------

# 27. Full Training Distribution

The complete training data contained:

``` text
Total samples: 754655
Classes: 51
```

Key counts:

``` text
0  no relationship       11,541   1.53%
47 watching              43,763   5.80%
50 worn by              375,269  49.73%
```

Almost half the entire training dataset belonged to:

``` text
worn by
```

This explains the model's learned shortcut:

``` text
Uncertain input
↓
Predict worn by
```

------------------------------------------------------------------------

# 28. Accuracy vs Macro F1

Original checkpoint:

``` text
Validation Accuracy : 0.6802
Macro F1            : 0.2759
Top-5 Accuracy      : 0.9574
```

The high accuracy was partly misleading because dominant classes
strongly influence accuracy.

Macro F1 revealed that many minority predicates performed poorly.

Some classes had F1 = 0.

The strongest classes included:

``` text
Class 50 : F1 0.8318
Class 47 : F1 0.8284
Class 14 : F1 0.7789
Class 29 : F1 0.6372
Class 19 : F1 0.6148
Class 21 : F1 0.5817
```

Several classes had zero performance.

------------------------------------------------------------------------

# 29. Confusion Analysis

Major confusions included:

``` text
True 50 → Pred 29 : 2090
True 29 → Pred 50 : 1087
True 50 → Pred 19 : 1078
True 27 → Pred 50 : 998
True 50 → Pred 21 : 830
```

This showed substantial overlap and confusion among frequent predicates.

------------------------------------------------------------------------

# 30. V2 Balanced Retraining

Because of the extreme class imbalance, a second training version was
created.

Objective:

``` text
Reduce dominant-class bias
Increase minority-class exposure
Improve macro F1
```

An imbalance-aware weighted sampler was added.

Concept:

``` text
Rare class
→ sampled more often

Frequent class
→ sampled less often
```

The original data was not deleted.

Only training sample selection was changed.

Validation remained unbalanced and representative:

``` text
Training:
Weighted sampling

Validation:
Normal loading
shuffle=False
```

------------------------------------------------------------------------

# 31. V2 Training Setup

The model architecture remained unchanged.

Fresh model:

``` python
model = CachedRelationshipModel(
    num_object_classes=150,
    num_predicate_classes=51,
    visual_dim=512,
    label_embedding_dim=64,
    geometry_dim=32,
    hidden_dim=256,
    dropout=DROPOUT,
).to(device)
```

Optimizer:

``` python
AdamW
```

Loss:

``` python
CrossEntropyLoss
```

Main balancing change:

``` text
Weighted training sampler
```

------------------------------------------------------------------------

# 32. V2 Result

Training completed:

``` text
V2 BALANCED TRAINING COMPLETE

Best epoch        : 2
Best Val Macro F1 : 0.2649
Checkpoint        : /kaggle/working/best_full_fusion_v2_balanced.pt
```

Comparison:

``` text
Original Macro F1 : 0.2759
V2 Macro F1       : 0.2649
```

The balanced version did not improve the chosen metric.

Therefore, the original 30-epoch checkpoint remained the current final
model.

------------------------------------------------------------------------

# 33. Final Inference Fix

The local testing notebook was corrected to explicitly handle:

``` text
class 0 = no relationship
```

This allowed the system to avoid assigning every pair a semantic
predicate.

Instead, it could distinguish:

``` text
actual relationship
```

from:

``` text
no relationship
```

------------------------------------------------------------------------

# 34. Final End-to-End Result

The final tested detection result produced:

``` text
Pairs with relationship    : 1
Pairs with no relationship : 9
```

Predicted relationship:

``` text
person → riding → bike (21.79%)
```

Pairs classified as no relationship:

``` text
person → car
person → car
person → bike
car → person
car → person
car → bike
bike → person
bike → person
bike → car
```

with explicit confidences generated by the model.

This is much more meaningful than the earlier output where nearly every
pair became `worn by`.

------------------------------------------------------------------------

# 35. Final Pipeline

The working system is now:

``` text
INPUT IMAGE
    ↓
YOLO OBJECT DETECTION
    ↓
Detected labels + boxes
    ↓
Visual Genome class mapping
    ↓
Convert to embedding IDs
    ↓
Generate compatible ordered pairs
    ↓
For each pair:
    ↓
Create union bounding box
    ↓
Crop union region
    ↓
Preprocess to 224 × 224
    ↓
Visual encoder
    ↓
512-D visual feature
    ↓
Calculate 8-D geometry
    ↓
CachedRelationshipModel
    ↓
51 logits
    ↓
0 = no relationship
1–50 = predicates
    ↓
SCENE GRAPH TRIPLETS
```

------------------------------------------------------------------------

# 36. Final Model Summary

``` text
Architecture:
CachedRelationshipModel
```

Inputs:

``` text
Visual feature          : 512
Subject embedding       : 64
Object embedding        : 64
Geometry feature        : 32
```

Fusion:

``` text
672 dimensions
```

Classifier:

``` text
672 → 256 → 51
```

Final checkpoint:

``` text
best_full_fusion_30epoch.pt
```

Best metrics:

``` text
Best epoch       : 7
Macro F1         : 0.2759
Accuracy         : 0.6802
Validation loss  : 1.4196
Top-5 accuracy   : 0.9574
```

------------------------------------------------------------------------

# 37. Problems Encountered

  -----------------------------------------------------------------------
  Problem                 Cause                   Solution
  ----------------------- ----------------------- -----------------------
  Model import failed     Package initializer     Simplified
                          imported unavailable    `__init__.py`
                          dependencies            

  Cached model missing    Model existed only in   Added it to local
  locally                 Kaggle notebook         `model.py`

  Filename said 15 epochs Old checkpoint filename Verified actual
                                                  30-epoch configuration

  `KeyError: 50`          Class mapping           Added class 0
                          incomplete              explicitly

  `worn by` everywhere    Extreme class imbalance Investigated training
                                                  distribution

  Suspected output        Incorrect hypothesis    Verified 51-class
  capping                                         argmax behaviour

  Label mismatch          Multiple ID systems     Added explicit
                                                  VG/embedding conversion

  V2 balancing            Weighted sampling       Retained original
  underperformed          insufficient            checkpoint
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 38. Main Technical Lessons

## Inspect class distributions early

The statistic:

``` text
49.73% of training samples = worn by
```

explained a major portion of the model behaviour.

## Accuracy alone is insufficient

A model can show:

``` text
68% accuracy
```

while having poor minority-class performance.

Macro F1 was essential.

## Index consistency is critical

The project uses several IDs:

``` text
YOLO class ID
Visual Genome object ID
Embedding index
Predicate dictionary ID
Model output class
```

These must never be mixed.

## End-to-end testing should be incremental

The progression used was:

``` text
Cached sample
↓
Ground-truth objects
↓
Ground-truth relationships
↓
YOLO detections
↓
Object pairs
↓
End-to-end relationship inference
```

This made debugging much easier.

------------------------------------------------------------------------

# 39. Current Status

The relationship component is operational.

It can:

``` text
1. Receive an image.
2. Use detected objects.
3. Convert labels to model-compatible IDs.
4. Generate object pairs.
5. Extract pair-level visual features.
6. Calculate geometry.
7. Fuse visual, semantic and spatial information.
8. Predict a predicate or no relationship.
9. Produce scene-graph-style triplets.
```

Current demonstrated result:

``` text
person → riding → bike
```

while unrelated pairs can be predicted as:

``` text
no relationship
```

------------------------------------------------------------------------

# 40. Final Conclusion

The relationship prediction system has successfully progressed from an
offline cached-feature training model into a functioning image-level
inference pipeline.

Complete journey:

``` text
Visual Genome Dataset
        ↓
Relationship preprocessing
        ↓
Cached 512-D features
        ↓
Kaggle dataset upload
        ↓
GPU training
        ↓
30-epoch full-fusion model
        ↓
Best checkpoint selection
        ↓
Local checkpoint restoration
        ↓
Cached validation testing
        ↓
Ground-truth image testing
        ↓
Union-region feature extraction
        ↓
YOLO object detection
        ↓
Object-pair generation
        ↓
Relationship / no-relationship prediction
        ↓
Scene graph triplets
```

The model still has limitations caused largely by severe predicate
imbalance in the training data. Nevertheless, the complete
infrastructure is now functioning, and the system has reached the
intended milestone:

``` text
IMAGE
  ↓
OBJECT DETECTION
  ↓
OBJECT PAIRS
  ↓
VISUAL + SEMANTIC + GEOMETRIC FUSION
  ↓
RELATIONSHIP / NO RELATIONSHIP
  ↓
SCENE GRAPH TRIPLETS
```

------------------------------------------------------------------------

# 41. Recommended Future Work

The current version should be preserved as the baseline.

Possible future improvements:

``` text
1. Class-weighted loss or focal loss.
2. Better imbalance strategies.
3. Predicate-specific confidence thresholds.
4. Top-k relationship predictions.
5. Improved object-pair proposal filtering.
6. Better YOLO-to-Visual-Genome mapping.
7. Full scene graph visualization.
8. Per-predicate evaluation.
9. More testing on unseen images.
```
