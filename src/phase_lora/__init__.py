"""Phase LoRA: a polar reparametrisation of the LoRA factors.

    dW = scale * Re(A B^H),   A = rho_a e^(i phi_a),  B = rho_b e^(i phi_b)

Same image as real LoRA at rank 2r and the same parameter count, so the
hypothesis class is untouched -- only the chart changes, and with it the path
gradient descent takes. Arithmetic is entirely real; dW merges into W.

An earlier `W^(1+BA)` exponent adapter, and the complex-residual-stream model
it needed, were removed.
"""

from .paths import MODEL_DIR, ROOT, RUNS_DIR
from .polar import PolarLoRALinear
from .stiefel import PhaseStiefel, PolarStiefel
from .train_utils import DATASET, TARGETS, attach, build_data, collate
from .unitary import OrthoLinear, batch_cayley, install_batched_cayley, n_params

__all__ = ["PolarLoRALinear", "PolarStiefel", "PhaseStiefel", "OrthoLinear", "n_params", "batch_cayley",
           "install_batched_cayley", "MODEL_DIR", "RUNS_DIR", "ROOT",
           "attach", "build_data", "collate", "DATASET", "TARGETS"]
