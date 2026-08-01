# Lesson 04 - Computer Vision: CNNs & Transfer Learning

> Read this with `src/convolution.py` open, and run `notebooks/04_computer_vision.ipynb` for the
> rendered diagrams (feature maps, augmentation, the transfer-learning schematic).

## 0. What you'll be able to do after this
- Explain how a computer "sees" an image and what a **convolution** computes.
- Explain **transfer learning** and the **two-stage** fine-tuning recipe.
- Run a from-scratch convolution (`python -m src.convolution`) and read the production
  fine-tuning code (`src/transfer_learning.py`).

## 1. The big picture (why this project exists)
You rarely train a vision model from zero - you have too few images and too little compute.
Instead you take a model **pretrained on millions of images** and adapt it to your task. That's
**transfer learning**, and it's how almost all real computer vision is done. To understand it, you
first need the one operation everything is built on: the **convolution**.

## 2. Foundations from scratch (the basics)
**An image is numbers.** Grayscale = a grid of brightnesses (0=black, 1=white). Colour = three
grids (Red, Green, Blue).

**Convolution.** Slide a tiny grid of weights (a **filter/kernel**, e.g. 3x3) across the image. At
each position, multiply the overlapping patch by the kernel and sum it. The result is a **feature
map**. `src/convolution.py` does exactly this in ~6 lines; an edge filter lights up at edges, a
blur filter smooths. (The demo asserts both behaviours.)

**CNN (Convolutional Neural Network).** Stack many convolution layers whose filters are **learned**
during training. Early layers learn edges; later layers learn shapes and objects. You don't design
the filters - gradient descent does.

**Transfer learning.** A pretrained CNN already has great filters. You **reuse the backbone** and
replace only the final **head** (the classifier) for your classes. Two stages:
1. **Freeze** the backbone, train only the new head (fast, stable).
2. **Unfreeze** everything and fine-tune with a **small** learning rate (gentle polish).
`src/transfer_learning.py` implements this with `timm` (a library of pretrained models).

**Data augmentation.** Randomly flip/crop/rotate training images. The label ("cat") doesn't change,
so the model learns to be robust - effectively free extra data.

## 3. How the pieces fit - the flow
```
                 (concept layer - runs on CPU)            (production layer - needs a GPU)
  image numbers --> conv2d + filters --> feature maps      images+augmentation
   (make_image)        (convolution.py)                          |
                                                                 v
                                              pretrained backbone (frozen) + NEW head
                                                                 |  stage 1: train head
                                                                 v  stage 2: unfreeze, tiny LR
                                                        fine-tuned model -> save -> HF Space demo
```

## 4. Code reading order
1. `src/convolution.py` - `conv2d`, the filters, and the demo (run it first).
2. `notebooks/04_computer_vision.ipynb` - see the feature maps + augmentation + schematic.
3. `src/transfer_learning.py` - the real two-stage fine-tune (`build_dataloaders`, `run_epoch`, `main`).
4. `app/app.py` - the Gradio demo you deploy as a Hugging Face Space.

## 5. Common mistakes (and how to avoid them)
- **Training from scratch on few images** -> overfits badly; use transfer learning + augmentation.
- **Too-large LR in stage 2** -> wrecks the pretrained weights; use ~1e-5.
- **Augmenting the validation set** -> inflates or distorts scores; only augment training data.
- **Forgetting normalization** -> match the preprocessing the backbone was pretrained with.

## 6. Check your understanding
1. What does a 3x3 edge kernel actually compute at one pixel?
2. Why retrain only the head first instead of the whole network?
3. Why is the stage-2 learning rate so much smaller than stage 1?
4. Why must augmentation be applied to training images but not validation images?

## 7. Mini-glossary (full versions in /GLOSSARY.md)
pixel, convolution, kernel/filter, feature map, CNN, backbone/head, transfer learning,
fine-tuning, data augmentation, ViT.

## 8. Going deeper
- timm (pretrained vision models): https://huggingface.co/docs/timm
- CS231n (convolutional networks) lecture notes: https://cs231n.github.io/
