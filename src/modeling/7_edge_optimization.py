"""
Phase 7: Edge AI Optimization.

Produces:
- dynamically quantized PyTorch model
- ONNX export for deployment simulation
- CPU inference benchmark JSON
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import onnxruntime as ort
import torch
from torch import nn

sys.path.append(str(Path(__file__).resolve().parents[1]))
from modeling.multimodal_fusion import ICUFusionModel  # noqa: E402


PROJECT_DIR = Path(r"C:\Users\rohit\MultiModal\icu_predictive_system")
MODEL_DIR = PROJECT_DIR / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def benchmark_torch(model: nn.Module, physio: torch.Tensor, acoustic: torch.Tensor, iterations: int) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            model(physio, acoustic)
        start = time.perf_counter()
        for _ in range(iterations):
            model(physio, acoustic)
        elapsed = time.perf_counter() - start
    return {
        "iterations": iterations,
        "total_seconds": elapsed,
        "mean_ms": elapsed * 1000.0 / iterations,
        "throughput_inferences_per_second": iterations * physio.shape[0] / elapsed,
    }


def benchmark_onnx(path: Path, physio: torch.Tensor, acoustic: torch.Tensor, iterations: int) -> dict[str, float]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = {
        "physio_seq": physio.numpy(),
        "acoustic_feat": acoustic.numpy(),
    }
    for _ in range(20):
        session.run(None, inputs)
    start = time.perf_counter()
    for _ in range(iterations):
        session.run(None, inputs)
    elapsed = time.perf_counter() - start
    return {
        "iterations": iterations,
        "total_seconds": elapsed,
        "mean_ms": elapsed * 1000.0 / iterations,
        "throughput_inferences_per_second": iterations * physio.shape[0] / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(MODEL_DIR / "swallow_flag_within_4h_fusion_model.pt"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    fp32_model = ICUFusionModel(physio_dim=8, acoustic_dim=128)
    fp32_model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    fp32_model.eval()

    quantized_model = torch.quantization.quantize_dynamic(
        fp32_model,
        {nn.Linear, nn.LSTM},
        dtype=torch.qint8,
    )

    fp32_path = MODEL_DIR / "edge_fp32_model.pt"
    int8_path = MODEL_DIR / "edge_int8_dynamic_model.pt"
    onnx_path = MODEL_DIR / "edge_fp32_model.onnx"
    metrics_path = MODEL_DIR / "phase7_edge_metrics.json"

    torch.save(fp32_model.state_dict(), fp32_path)
    torch.save(quantized_model.state_dict(), int8_path)

    physio = torch.randn(args.batch_size, 1, 8)
    acoustic = torch.randn(args.batch_size, 128)
    torch.onnx.export(
        fp32_model,
        (physio, acoustic),
        onnx_path,
        input_names=["physio_seq", "acoustic_feat"],
        output_names=["risk_score"],
        dynamic_axes={
            "physio_seq": {0: "batch", 1: "seq_len"},
            "acoustic_feat": {0: "batch"},
            "risk_score": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    metrics = {
        "source_checkpoint": str(checkpoint),
        "fp32_model_size_mb": file_size_mb(fp32_path),
        "int8_dynamic_model_size_mb": file_size_mb(int8_path),
        "onnx_model_size_mb": file_size_mb(onnx_path),
        "torch_fp32": benchmark_torch(fp32_model, physio, acoustic, args.iterations),
        "torch_int8_dynamic": benchmark_torch(quantized_model, physio, acoustic, args.iterations),
        "onnxruntime_fp32": benchmark_onnx(onnx_path, physio, acoustic, args.iterations),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"PHASE 7 COMPLETE metrics={metrics_path}")


if __name__ == "__main__":
    main()
