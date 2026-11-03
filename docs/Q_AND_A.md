# Project Q&A preparation

## Why did we use Django and React?

Django provides a clean backend API for image upload, preprocessing, model loading, and prediction. React provides a responsive user interface where users can upload images, enter metadata, and understand results.

## Why did we use transfer learning?

The dataset is not large enough to train a very deep CNN from scratch reliably. Transfer learning uses visual features learned from a large image dataset and adapts them to the skin lesion task.

## Why EfficientNetB3?

Both classifiers use an ImageNet-pretrained EfficientNetB3 backbone with 300 by 300 inputs. This keeps the visual backbone consistent while comparing the two implemented heads. Alternative backbones and inference latency have not been benchmarked in this experiment.

## Why patient-level splitting?

Some patients or lesions may have multiple images. If random image splitting is used, images from the same patient can appear in both training and test sets, causing data leakage and overly optimistic results.

## Why do we use more than accuracy?

Medical classification needs sensitivity/recall, precision, F1-score, confusion matrix, and per-class metrics. Accuracy alone can hide poor performance on minority or dangerous classes.

## Why add metadata?

We compare image-only classification with fusion of five fields available before diagnosis: age, gender, body region, smoking and drinking history. Their usefulness must be assessed from the saved evaluation results. Biopsy status is excluded because it can encode the diagnostic process.

## Why Grad-CAM?

The code includes Grad-CAM utilities for visualizing class-sensitive image regions. Validated heatmap examples are outside this focused comparison; they should not be presented as established clinical or causal explanations.

## Why image quality warnings?

Bad images can reduce prediction reliability. The app checks brightness, blur, and resolution to warn users before interpreting predictions.

## Why is there a medical disclaimer?

The model is educational and not clinically validated. A professional system must not claim to diagnose disease without medical approval, safety testing, and regulatory validation.

## What contribution does this project demonstrate?

The project uses established methods. Its contribution is a documented comparison with patient-disjoint splits, realistic input restrictions, class-level evaluation, error examples, and a working web prototype. Explain the limitations and your personal contribution accurately.
