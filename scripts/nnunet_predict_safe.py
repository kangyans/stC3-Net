"""Run as: python scripts/nnunet_predict_safe.py -i IN -o OUT -t 27 -m 2d
(with nnU-Net v1 installed and nnUNet_raw_data_base / nnUNet_preprocessed /
RESULTS_FOLDER set; Task027_ACDC from zenodo.org/records/4003545).

nnU-Net v1 inference with torch.load(weights_only=True) kept on: only the
numpy types stored in the official Task027_ACDC checkpoints are allow-listed
(torch.serialization.add_safe_globals), as PyTorch's error message advises."""
import sys
import numpy as np
import torch
from torch.serialization import add_safe_globals
add_safe_globals([np.core.multiarray.scalar, np.dtype,
                  type(np.dtype('float64')), type(np.dtype('float32')),
                  type(np.dtype('int64'))])
from nnunet.inference.predict_simple import main
sys.argv = ['nnUNet_predict'] + sys.argv[1:]
main()
