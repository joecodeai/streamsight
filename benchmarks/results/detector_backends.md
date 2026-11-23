Model `yolo11n.onnx` at 640x640, forward pass only. GPU: NVIDIA GeForce RTX 5090. CPU: Intel(R) Core(TM) Ultra 9 285K. ONNX Runtime 1.32.0, OpenVINO 2026.4.1. Measured 2026-11-23.

| Backend | Batch | p50 latency (ms) | p95 latency (ms) | Throughput (img/s) |
|---|---:|---:|---:|---:|
| ort-cpu | 1 | 11.25 | 14.13 | 87 |
| ort-cpu | 4 | 37.97 | 45.86 | 103 |
| ort-cpu | 8 | 86.25 | 120.49 | 87 |
| ort-cuda | 1 | 2.10 | 2.18 | 475 |
| ort-cuda | 4 | 4.20 | 4.44 | 947 |
| ort-cuda | 8 | 7.79 | 8.03 | 1022 |
| ort-tensorrt | 1 | 1.11 | 1.18 | 891 |
| ort-tensorrt | 4 | 2.78 | 2.99 | 1438 |
| ort-tensorrt | 8 | 5.26 | 5.37 | 1518 |
| openvino-cpu | 1 | 11.59 | 13.01 | 87 |
| openvino-cpu | 4 | 38.70 | 48.39 | 100 |
| openvino-cpu | 8 | 76.11 | 81.33 | 104 |
| trtexec-fp32 | 1 | 0.72 | 0.91 | 1361 |
| trtexec-fp32 | 4 | 1.33 | 1.68 | 2893 |
| trtexec-fp32 | 8 | 2.00 | 2.40 | 3917 |
| trtexec-fp16 | 1 | 0.39 | 0.40 | 2530 |
| trtexec-fp16 | 4 | 0.61 | 0.61 | 6570 |
| trtexec-fp16 | 8 | 0.93 | 0.94 | 8609 |
