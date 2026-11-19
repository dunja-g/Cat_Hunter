# Prototype: Cat vs. Non-Cat Classification (2 classes)

This folder holds the **early 2-class prototype** (cat vs. non-cat) that came before the
final 5-breed classifier. It is kept for reference only. The plots and results below are
from this prototype, **not** from the final model in [`training_src/`](../training_src/).
The 2-class code (`baseline_model.py`, `transfer_model.py`, ...) was later adapted to 5 classes in `training_src/`.

| File | Description |
| :--- | :--- |
| `cat_transfer_learning_project.py` | All-in-one notebook-style script (Kaggle / Colab / Jupyter / VS Code) |
| `kaggle_cat_not_cat_complete.py` | Cleaned-up Kaggle version of the same pipeline (ResNet50) |
| `plots/` | Training curves and confusion matrices of the baseline CNN and the transfer model |

*(Translated from the original Chinese project plan.)*

---

## Goal

Build an image classification pipeline covering data preprocessing and augmentation, a
baseline convolutional neural network (CNN), an advanced model trained with transfer
learning, hyperparameter tuning, and final evaluation and metric analysis.

## Team roles

The work was split between **Person A (Data Engineer / Model Developer)** and
**Person B (Augmentation Specialist / Hyperparameter Optimizer)**.

| Phase | Module | Person A (data and baseline model) | Person B (augmentation and transfer learning) |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Data preprocessing and augmentation** | Data cleaning and split script (resize, normalise, 70/15/15 split) | Data augmentation (rotation, flips, brightness, ...) |
| **Phase 2** | **Build the models** | Set up the environment; build a lightweight baseline CNN from scratch | Load a pretrained model (MobileNetV2 / ResNet50) for transfer learning |
| **Phase 3** | **Training and hyperparameter tuning** | Run the training loop; monitor train/val loss to prevent overfitting | Try hyperparameter combinations (learning rate, batch size <= 32, Adam/SGD) |
| **Phase 4** | **Evaluation and metrics** | Test the final models together: accuracy, confusion matrix, precision, recall | Test the final models together: accuracy, confusion matrix, precision, recall |

## Phase plan

### Phase 1: Data preprocessing and augmentation
- **Goal:** clean and prepare about 4,000 images and use augmentation to reduce overfitting.
- Resize all images to 224 × 224 pixels.
- Normalise pixel values to [0, 1].
- Randomly split into **70% train, 15% validation, 15% test**.
- Augmentation: random rotation, horizontal/vertical flips, brightness adjustments.

### Phase 2: Baseline and transfer model
- **Goal:** get a working baseline quickly, then compare it with a pretrained model.
- Install a deep learning framework (PyTorch recommended; TensorFlow/Keras also possible).
- **Baseline model (Person A):** a minimal CNN (e.g. 3 conv layers + max pooling + fully connected layer) to make sure the whole pipeline runs end to end.
- **Transfer model (Person B):** load a pretrained **MobileNetV2** or **ResNet50**, freeze the early feature-extraction layers, and replace and train only the classification head.

### Phase 3: Training and hyperparameter tuning
- **Goal:** train efficiently and tune the model and its parameters.
- Write a full training loop (forward pass, loss, backward pass, optimiser step).
- Log and plot train/validation loss and accuracy to diagnose overfitting or underfitting.
- Hyperparameter experiments:
  - **Learning rate:** different initial values, or a learning-rate decay schedule.
  - **Batch size:** 32 or below, to avoid running out of memory on a laptop.
  - **Optimiser:** compare convergence speed and final results of Adam and SGD.

### Phase 4: Evaluation and metrics
- **Goal:** evaluate the final models on a held-out test set and report their performance.
- Test both the baseline and the transfer model on the test set.
- Compute **accuracy**, **precision**, **recall** and **F1 score**.
- Plot a **confusion matrix** to see which classes get confused (e.g. which non-cat images are predicted as cats).

---

## Prototype results

All phases were completed. End-to-end training and evaluation used a 10% subset of a
cat vs. boat dataset (1,500+ images; 228 test images).

| Model | Test accuracy | Training setup | Details |
| :--- | :--- | :--- | :--- |
| **Baseline CNN (Person A)** | **97.37%** | 3 epochs (CPU) | 6 errors (4 boats predicted as cat, 2 cats predicted as boat) |
| **Transfer learning, ResNet (Person B)** | **100.00%** | 3 epochs (CPU) | 0 errors |

**Conclusion:** even with little data, few epochs and CPU-only training, transfer learning
with a pretrained backbone classified the test set perfectly and clearly beat the baseline
CNN trained from scratch.
