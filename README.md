# Oral Disease Image Classification

End-to-end deep learning project that classifies oral images into six categories: **Calculus, Caries, Gingivitis, Hypodontia, Mouth Ulcer, Tooth Discoloration**. It includes a custom CNN, three pretrained models, hyperparameter tuning, a comparative analysis, and a Gradio app that serves the best model selected on validation data.

![Gradio demo](docs/gradio_demo.png)

## Highlights

- **Duplicate-aware data preparation:** exact (MD5) and near-duplicate (orientation-aware dHash) images are grouped, groups with conflicting labels are removed, and the train/validation/test split is done **by group** so no copy of a training image appears in validation or test.
- **Custom CNN** (residual network trained from scratch) as a baseline.
- **Three pretrained models:** ResNet50 (with hyperparameter tuning), EfficientNet-B0, ConvNeXt-Tiny.
- **Efficient training:** images cached once as uint8 tensors, GPU-side augmentation (affine, flip, mild color jitter, CutMix), AdamW with warmup + cosine decay, mixed precision, label smoothing, class-balanced sampling, flip test-time augmentation.
- **Deployment:** Gradio app using the model selected on the validation set.

## Dataset

Oral Diseases dataset on Kaggle (`salmansajid05/oral-diseases`), 5,563 images in 6 classes.

| Stage | Images / groups |
|---|---|
| Raw images | 5,563 |
| Removed (conflicting-label groups) | 1,966 images in 126 groups |
| Clean images retained | 3,597 in 1,402 unique groups |
| Training | 2,452 images (980 groups) |
| Validation | 211 groups (one representative per group) |
| Test | 211 groups (one representative per group) |

## Results

Model selection was done on the validation set only. The test set is used for final reporting.

| Model | Val accuracy | Val macro-F1 | Test accuracy | Test macro-F1 |
|---|---|---|---|---|
| **EfficientNet-B0 (selected, deployed)** | **0.938** | **0.922** | 0.900 | 0.870 |
| ResNet50 (tuned) | 0.929 | 0.912 | **0.929** | **0.912** |
| ConvNeXt-Tiny | 0.924 | 0.910 | 0.915 | 0.893 |
| Custom CNN | 0.777 | 0.741 | 0.834 | 0.802 |

Test accuracy of the deployed model: 0.900 (95% bootstrap CI 0.858 to 0.938), macro-F1 0.870.

Per-class results of the deployed model on the test set:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Calculus | 0.762 | 0.842 | 0.800 | 19 |
| Caries | 0.800 | 0.741 | 0.769 | 27 |
| Gingivitis | 0.909 | 0.980 | 0.943 | 51 |
| Hypodontia | 0.950 | 0.983 | 0.966 | 58 |
| Mouth Ulcer | 0.939 | 0.939 | 0.939 | 33 |
| Tooth Discoloration | 0.941 | 0.696 | 0.800 | 23 |

### Hyperparameter tuning (ResNet50)

Three configurations were compared on validation data. The best one used the highest learning rate and the strongest regularization.

| Trial | lr | head lr | weight decay | label smoothing | Val accuracy | Val macro-F1 |
|---|---|---|---|---|---|---|
| 2 (best) | 3e-4 | 1.5e-3 | 0.05 | 0.10 | 0.934 | 0.922 |
| 1 | 1.5e-4 | 1e-3 | 0.02 | 0.05 | 0.915 | 0.897 |
| 3 | 1e-4 | 8e-4 | 0.01 | 0.00 | 0.891 | 0.870 |

### Discussion

- All three pretrained models clearly outperform the custom CNN trained from scratch, which shows the benefit of transfer learning on a limited dataset.
- The pretrained models are close to each other. EfficientNet-B0 was selected because it had the best validation accuracy and log-loss, and it is also the smallest and fastest model (checkpoint about 15.6 MB), which suits deployment.
- Classes with more training images (Hypodontia, Gingivitis) are recognized best. Tooth Discoloration has the lowest recall (0.70) and is most often predicted as Caries.

## Run it

1. Open the notebook on Kaggle, attach the dataset, and enable a GPU and Internet (needed to download ImageNet weights).
2. Run all cells. Full run: about 23 minutes on a Tesla T4.
3. The last cell launches the Gradio app. It runs locally and can also create a temporary public link.

Install locally:

```bash
pip install -r requirements.txt
```

## Repository structure

```
Oral_Diseases_Image_Classification.ipynb   # full pipeline: data, training, evaluation, Gradio app
requirements.txt
docs/gradio_demo.png                        # screenshot of the app
```

## Disclaimer

Educational image-classification prototype. It is not a medical device and must not be used for diagnosis.
