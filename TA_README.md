# TA Instructions & Notes

Hello! Thanks for reviewing our submission. Because we ran into issues with the KAIST KCloud credentials we trained our model on Google Colab. Everything you need to reproduce our results locally (or in Colab) is summarised here.

## Repository Structure & Required Files

- `custom_model.py`, `train.py`, `dataset.py`, and `src/` contain the full training and sampling implementation.
- `requirements.txt` is the minimal dependency list used during training.
- `checkpoints/best_model.pt` and `checkpoints/model_config.json` are the artefacts produced by our best run (≈20 k iterations so far).
- Optional logs and samples live under `results/` and `samples/` if you want quick reference visuals.

## Reproducing Training (Colab Workflow)

1. **Create a fresh GPU Colab runtime** and mount Drive:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```
2. **Copy the repo into the runtime** (adjust the Drive path as needed):
   ```python
   import shutil, os
   src = "/content/drive/MyDrive/your/path/Diffusion-2025-Image_Challenge"
   dst = "/content/Diffusion-2025-Image_Challenge"
   if os.path.exists(dst):
       shutil.rmtree(dst)
   shutil.copytree(src, dst)
   %cd /content/Diffusion-2025-Image_Challenge
   ```
3. **Install dependencies**:
   ```python
   !pip install kagglehub
   !pip install -r requirements.txt
   # Optional: match our torch build
   # !pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
   ```
4. **Launch training** (this will auto-download the Simpsons dataset via `kagglehub`):
   ```python
   !python train.py --device cuda --batch_size 32 --num_iterations 100000
   ```
   - Checkpoints and the config JSON are emitted under `checkpoints/`.
   - Intermediate checkpoints and preview samples are saved every 10 k iterations in `results/<timestamp>/`.

## Using the Provided Checkpoint

- The current best snapshot (`checkpoints/best_model.pt`) corresponds to ~10 k iterations.  
- To sample:
  ```bash
  python sampling.py --ckpt_path checkpoints/best_model.pt \
                     --save_dir samples/best_model \
                     --num_samples 1000 --batch_size 64 \
                     --nfe_list 1 2 4 --device cuda
  ```
- To evaluate FID (optional—official evaluation will replace these scripts):
  ```bash
  python measure_fid.py --generated_dir samples/best_model/nfe=1
  python measure_fid.py --generated_dir samples/best_model/nfe=2
  python measure_fid.py --generated_dir samples/best_model/nfe=4
  ```

## Notes & Known Limitations

- Because of the KCloud credential issue all experiments/training were executed on Colab; no environment-specific paths should remain in the code.
- The current checkpoint is early in training. Longer runs (continuing `train.py` with the same command) should keep improving the loss/FID.
- We added a `torch.no_grad()` guard around the sample previews generated during training (see `train.py`) to prevent GPU OOMs at checkpoint time.

Please reach out if anything fails to run—we made sure `train → sample → evaluate` works end-to-end in a clean Colab runtime. The submission does **not** ship with a pre-built environment or virtualenv; please install dependencies as outlined above when reproducing the results. Thank you!  
