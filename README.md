# 3D-Dogs: Silhouette-Guided 3D Reconstruction of Dogs from Single Images

This repository builds on the **3D-Fauna** codebase (Li *et al.*, CVPR 2024) to specifically tackle the challenges of reconstructing accurate 3D dog meshes from a single RGB image. Dogs exhibit vast pose and breed variation—and viewpoints like top-down or frontal shots are underrepresented in typical training sets—leading to distorted outputs (e.g., flattened underbellies). We introduce a **mask-based adversarial discriminator** and a lightweight multiple-hypothesis selection scheme to enforce silhouette consistency from unseen angles.

Seonghee Lee(shlee@cs.stanford.edu) - CS 231A

---

## Method Overview

### 1. Base 3D Reconstruction Pipeline (3D-Fauna)

At its core, we retain the standard 3D-Fauna architecture:

1. **Prior Shape Predictor**
   A shared “bank” of shape embeddings (`netBase`) encodes coarse SDF-based animal priors.
2. **Instance Predictor**
   Given a single input image `I`, the instance network (`netInstance`) predicts:

   * A refined 3D shape as a tetrahedral SDF mesh.
   * An articulation parameter set (pose).
   * A texture field for albedo.
   * Camera parameters (rotation hypotheses, translation, focal length, etc.).
3. **Differentiable Renderer**
   The predicted shape + texture + camera yields rendered outputs:

   * **RGB image** (reconstructed appearance).
   * **Silhouette mask** (foreground binary).
   * **DINO feature maps** (optional perceptual features).
     These are compared against the ground-truth image, mask, and DINO features to compute pixel-wise and perceptual losses.

Despite strong performance on generic quadrupeds, directly applying this pipeline to dog images often fails when dogs are photographed from uncommon angles (e.g., overhead). The network simply “hallucinates” a plausible—but inaccurate—underbelly, since it has never seen real silhouettes from that viewpoint during training.

---

### 2. Mask Discriminator: Enforcing Silhouette Realism from Unseen Views

#### 2.1 Motivation

A 3D generator trained solely on input-view losses can cheat by producing a shape that matches the mask from the observed camera but collapses or warps under novel rotations. Enforcing pixel-level consistency on a random rotated silhouette—with only a reconstruction loss—still does not teach the network what a “correct” top-down dog mask should look like. Instead, we introduce a small CNN discriminator (`netDisc`) that learns real silhouette statistics (across all breeds/poses) and pushes the generator to produce silhouettes that “fool” it.

#### 2.2 Pipeline

1. **Input-View Silhouette**

   * From batch `I`, we already compute a binary **mask\_gt** (ground truth) and let the model produce its **mask\_pred\_iv** under the predicted camera.

2. **Random-View Silhouette**

   * We sample a **random rotation matrix** `R_rand` (either yaw-only or full 3D).
   * We update the predicted 3D shape via `netInstance`, but replace its camera’s rotation matrix with `R_rand`.
   * We render this rotated shape (same texture, same articulation) to obtain **mask\_pred\_rv**.

3. **Class Embedding Concatenation**

   * We embed the instance’s predicted bank code (the “class vector”) into a `1×H×W` feature map and concatenate it with each binary silhouette, forming a tensor of shape `(1 + C)×H×W` (where `C` is the size of the class embedding).

4. **Discriminator Forward Pass**

   * `D_real`: feed `mask_gt ‖ class_embed` (concatenate along channels) → expect **Real** label.
   * `D_fake_iv`: feed `mask_pred_iv ‖ class_embed` → expect **Fake** (for discriminator’s loss).
   * `D_fake_rv`: feed `mask_pred_rv ‖ class_embed` → expect **Fake**.

5. **Adversarial Losses**

   * **Discriminator Loss** (BCE):

     $$
       \mathcal{L}_D = \tfrac{1}{3}\bigl[\,
         -\log D(\,mask_{gt} \| class\,)
         - \log\bigl(1 - D(\,mask_{pred\_iv} \| class\,)\bigr)
         - \log\bigl(1 - D(\,mask_{pred\_rv} \| class\,)\bigr)
       \bigr].
     $$

     We also optionally add a small gradient penalty on real masks (if `disc_gt=True`).
   * **Generator (3D Predictor) Loss** (BCE):

     $$
       \mathcal{L}_{G\_adv} 
       = \tfrac{1}{2}\Bigl[
         -\log D(\,mask_{pred\_iv} \| class\,) 
         \;-\;\log D(\,mask_{pred\_rv} \| class\,)\Bigr].
     $$

     In other words, the generator tries to push both input-view and random-view silhouettes to be classified as Real.

6. **Total Generator Loss**

   $$
     \mathcal{L}_G 
     = \underbrace{\mathcal{L}_\text{RGB} + \lambda_\text{mask}\,\mathcal{L}_\text{mask} + \dots}_{\text{original reconstruction losses}}
     \;+\;\alpha\,\mathcal{L}_{G\_adv}.
   $$

   * We keep the original RGB-, mask-, and DINO-feature reconstruction terms.
   * We add the adversarial term with weight `α = mask_disc_loss_weight` (e.g., 0.2).

During backprop:

1. **Generator Step**: we backpropagate through `netInstance`, `netBase`, and the differentiable renderer, summing the reconstruction + adversarial generator loss.
2. **Discriminator Step**: we freeze the generator, then backpropagate through `netDisc` to minimize `ℒ_D`. This two-step loop is standard GAN training.

---

### 3. Random-View Sampling

* **Yaw-Only Sampling** (baseline): draw a random yaw angle in `[0, 2π)`.
* **Full 3D Sampling** (optional): sample a uniform random unit quaternion (pitch/yaw/roll).

  * When `cfg_render.uniform_3d_rotation=True`, we generate a random 3×3 rotation matrix from quaternions. Otherwise, we stick to Y-axis rotations for silhouette diversity.
* **Camera Z-Offset**: by increasing `cam_pos_z_offset` (e.g., from 10→15), we ensure that the entire animal fits within the viewport even when rotated to extreme elevations (e.g., top-down).

---

### 4. Post-Processing: ViewpointBiasDetector

Even with silhouette adversarial training, extremely degenerate input viewpoints (e.g., pure overhead with severe occlusion) can still produce ambiguous results. We implement a lightweight **ViewpointBiasDetector** during inference:

1. **Estimate Camera Direction**
   After `netInstance` predicts its camera → we compute the forward‐looking direction in world space.
2. **Compute Similarity to “Top-Down” Normal**
   We compare the camera’s “downward” axis to the world’s up vector: if the absolute cosine similarity is below a threshold (e.g., 0.25), we suspect a degenerate viewpoint.
3. **Multiple-Hypothesis Ranking**

   * We generate *multiple* random rotations (e.g., 5–10 candidate yaw/pitch combinations).
   * For each candidate, render the silhouette and compute the discriminator’s “realness” score.
   * We also compute a lightweight anatomical plausibility measure: e.g., the variance of silhouette height, or silhouette aspect ratio.
   * We reproject the input photo’s visible mask into 3D (via back-projection) and measure 2D consistency with each hypothesis’s mask.
   * We choose the hypothesis that maximizes a weighted sum of (Discriminator score + silhouette‐input consistency).
4. **Final Mesh**
   We output the mesh corresponding to the selected hypothesis. By explicitly reranking over multiple views, we further reduce catastrophic shape collapse under rare angles—without retraining the entire network.

---

## Folder Structure & Data

```
3DAnimals/
├── config/                       # Hydra configurations (train/test/infer)
├── model/
│   ├── datasets/
│   ├── models/
│   │   ├── AnimalModel.py        # Base classes, rendering, losses, etc.
│   │   ├── Fauna.py              # 3D-Fauna extensions: mask discriminator, etc.
│   │   └── … 
│   ├── render/                   # Differentiable renderer utilities
│   └── predictors/               # Base / instance predictor definitions
├── results/
│   └── fauna/pretrained_fauna/   # Inference outputs for various dog breeds
│       └── <breed_name>/         # e.g., “weimaraner_dog”
│           ├── visualization/    # Rendered images (input, mask_gt, mask_pred, mesh.obj, poses, etc.)
│           ├── pretrained_fauna.pth  (removed from Git—see instructions below)
│           └── … 
├── run.py                        # Main training / testing entry point (Hydra)
└── README.md                     # This file
```

### Download Pretrained Models & Results

* **Pretrained checkpoint** (160 MB) and **visualization outputs** are hosted via Google Drive:
  👉 [Drive Folder](https://drive.google.com/drive/folders/11BmuqdPvFwrn9AUAFV2g1CxjemZKP0p_)
  Download and unzip into:

  ```
  3DAnimals/results/fauna/pretrained_fauna/
  ```

  so that you have:

  ```
  results/fauna/pretrained_fauna/pretrained_fauna.pth
  results/fauna/pretrained_fauna/<breed_name>/visualization/…
  ```


