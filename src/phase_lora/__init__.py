"""Phase LoRA: adapter factors stored in polar coordinates.

    dW = scale * Re(A B^H),   A = rho_a e^(i phi_a),  B = rho_b e^(i phi_b)

The function class and the parameter count equal real LoRA's at rank 2r; only the coordinates that the optimiser
updates differ. The arithmetic is real, and dW merges into W.
"""

from .paths import DATA_DIR, MODEL_DIR, ROOT, RUNS_DIR

__all__ = ["DATA_DIR", "MODEL_DIR", "RUNS_DIR", "ROOT"]
