"""Vendored minimal subset of the SR3-style diffusion codebase.

Source: ReDDiT — "Efficient Diffusion as Low Light Enhancer" (CVPR 2025,
arXiv:2410.12346), itself based on SR3 — Saharia et al., "Image
Super-Resolution via Iterative Refinement" (CVPR 2022).

Only the pieces needed to build and run the conditional UNet +
GaussianDiffusion are kept:

    networks.py                  define_G() factory
    ddpm_modules/diffusion.py    GaussianDiffusion (schedules, q_sample, ...)
    ddpm_modules/unet.py         conditional UNet

Modification relative to the upstream files: unused training-time imports
(cv2 / scikit-learn / lpips) were removed so that inference carries no
dependency on them. No functional code was changed.
"""