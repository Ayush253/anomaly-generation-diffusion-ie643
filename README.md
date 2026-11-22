# MVTec-AD Anomaly Generation

An IE-643 course project for generating synthetic defects on MVTec-AD industrial images. It includes a diffusion-model baseline, category-specific adapter training notebooks, and a Gradio app for generating anomalies on bottle and capsule images.

## Project files

- `baseline_good.ipynb` — trains the normal-image diffusion inpainting model.
- `adapter_bottle_train.ipynb` — trains the bottle adapter and empirical mask bank.
- `adapter_capsule_train.ipynb` — trains the capsule adapter and empirical mask bank.
- `anomaly_ui_v2_app.py` — launches the interactive anomaly-generation app.

## Setup

Use Python with PyTorch installed for your hardware (CPU or CUDA), then install the project dependencies:

```bash
python -m pip install "diffusers==0.30.3" "gradio==5.49.1" pillow "numpy==2.0.2" "huggingface-hub>=0.34.0,<1.0" scikit-learn seaborn matplotlib
```

The training notebooks expect the MVTec-AD dataset and a trained normal-model checkpoint. Set the data and checkpoint paths in the notebook configuration cells before running training.

## Run the app

The app requires three `.pt` checkpoint files for each category you want to load: the normal diffusion checkpoint, the Stage 3 adapter, and the empirical mask bank. Place them in category-specific folders under `weights/`, using the expected filenames:

```text
weights/
  bottle/
    mvtec_bottle_diffusion_inpaint_improved.pt
    type_conditioned_adapter_v3.pt
    generalized_empirical_mask_bank_v3.pt
  capsule/
    mvtec_capsule_diffusion_inpaint_improved.pt
    type_conditioned_adapter_v3.pt
    generalized_empirical_mask_bank_v3.pt
```

Start the app with both categories, or select one:

```bash
python anomaly_ui_v2_app.py --weights-root ./weights --category auto
python anomaly_ui_v2_app.py --weights-root ./weights --category bottle
```

Open the local URL printed by Gradio (default port `7860`). The app lets you upload a normal image, choose a defect type and severity, and view generated images alongside their exact masks.
