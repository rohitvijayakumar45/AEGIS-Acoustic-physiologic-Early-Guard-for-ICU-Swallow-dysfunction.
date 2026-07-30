import torch
import torch.nn as nn
import onnx
import onnxruntime
import time
import os
import psutil
import numpy as np
import json
from pathlib import Path
from baseline_models import LSTMModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    models_dir = Path(r"c:\Users\rohit\MultiModal\AEGIS\data\models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "lstm_baseline.pt"
    onnx_path = models_dir / "lstm_baseline.onnx"
    
    logger.info(f"Load lstm_baseline.pt from {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = torch.load(model_path)
    model.eval()
    model = model.cpu()
    
    # Dummy input based on common sequential data shapes (batch, seq_len, features)
    sequence_length = 24
    input_size = model.lstm.input_size if hasattr(model, 'lstm') else 34
    dummy_input = torch.randn(1, sequence_length, input_size)
    
    # Export to ONNX
    logger.info("Export to ONNX.")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'], 
        dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'}, 'output': {0: 'batch_size'}}
    )
    
    # Verify numeric consistency
    logger.info("Verify numeric consistency.")
    
    with torch.no_grad():
        pytorch_pred = model(dummy_input).numpy()
    
    ort_session = onnxruntime.InferenceSession(str(onnx_path))
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    onnx_pred = ort_session.run(None, ort_inputs)[0]
    
    diff = (pytorch_pred - onnx_pred)
    max_diff = diff.max()
    if np.max(np.abs(diff)) < 1e-5:
        logger.info(f"Numeric consistency verified. Max diff: {np.max(np.abs(diff))}")
    else:
        logger.error(f"Numeric consistency failed. Max diff: {np.max(np.abs(diff))}")
        
    # Log metrics
    logger.info("Log metrics.")
    
    # Inference latency
    start = time.time()
    for _ in range(100):
        _ = ort_session.run(None, ort_inputs)
    end = time.time()
    latency = (end - start) / 100 * 1000 # ms
    
    # Throughput
    throughput = 1000 / latency # seq/sec
    
    # Model size
    model_size = os.path.getsize(onnx_path) / (1024 * 1024) # MB
    
    # Peak memory
    process = psutil.Process(os.getpid())
    peak_memory = process.memory_info().rss / (1024 * 1024) # MB
    
    logger.info(f"Inference latency: {latency:.2f} ms")
    logger.info(f"Throughput: {throughput:.2f} seq/sec")
    logger.info(f"Model size: {model_size:.2f} MB")
    logger.info(f"Peak memory: {peak_memory:.2f} MB")

if __name__ == "__main__":
    import numpy as np
    main()
