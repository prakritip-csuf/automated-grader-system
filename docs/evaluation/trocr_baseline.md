# Milestone 1 — TrOCR Handwriting Recognition Baseline

Objective: Implement and evaluate a handwriting recognition pipeline using Microsoft's pretrained TrOCR model.

Implementation: The system uses OpenCV for handwriting line segmentation and TrOCR for text recognition.

The pipeline consists of:
- Image preprocessing and line segmentation
- Recognition of individual handwritten lines using TrOCR
- Combining recognized lines into a complete answer
- Returning recognition results through a FastAPI endpoint

Evaluation:

- Dataset: IAM Handwriting Database (test subset)

- Model: microsoft/trocr-base-handwritten

- Metrics: Character Error Rate (CER), Word Error Rate (WER)

Limitations:

The TrOCR checkpoint is designed primarily for individual handwritten text lines. Automatic line segmentation may fail when handwritten lines overlap, spacing is inconsistent, or image backgrounds contain shadows.

The IAM dataset is also closely related to the model's fine-tuning data. Therefore, results on IAM should not be interpreted as independent evidence of generalization to new handwriting styles.

Conclusion:

The pretrained TrOCR model has been integrated into the FastAPI application and evaluated using CER and WER. The baseline provides the handwriting recognition component for the subsequent semantic grading pipeline.