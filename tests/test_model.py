import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import DPT


def test_forward_pass_shape():
    model = DPT(decoder_channels=256)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 224, 224), f"Expected (2,1,224,224), got {out.shape}"


def test_forward_pass_cpu():
    """Model works on CPU without any .cuda() errors."""
    model = DPT(decoder_channels=128)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 224, 224)


def test_output_non_negative():
    """Output should be non-negative due to final ReLU activation."""
    model = DPT(decoder_channels=256)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert (out >= 0).all(), "Depth output contains negative values"


def test_different_decoder_channels():
    """Model works with different decoder channel sizes."""
    for channels in [64, 128, 256]:
        model = DPT(decoder_channels=channels)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 224, 224), f"Failed for channels={channels}"
