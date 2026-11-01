### Dynamic instructions per inference (mean over images)

| Config | Description | Normal model | High-sparsity model | vs B0 (normal / sparse) | vs X2 (normal / sparse) | .text bytes |
|---|---|---:|---:|---:|---:|---:|
| B0 | naive dense, RV64IMC | 13,350.7 M | 13,352.3 M | 1.0x / 1.0x | 0.04x / 0.04x | 9372 |
| B1 | optimized dense, RV64IMC | 5,428.6 M | 5,430.2 M | 2.5x / 2.5x | 0.10x / 0.10x | 9770 |
| X1 | dense + xqmac8 | 586.2 M | 587.8 M | 22.8x / 22.7x | 0.94x / 0.94x | 11282 |
| X2 | dense + xqmac8 + xrequant | 553.8 M | 553.6 M | 24.1x / 24.1x | 1.00x / 1.00x | 8140 |
| S0 | sparse-input (CSR), RV64IMC | 2,907.0 M | 2,310.1 M | 4.6x / 5.8x | 0.19x / 0.24x | 10244 |
| S1 | sparse + xsqmac8 | 844.2 M | 785.2 M | 15.8x / 17.0x | 0.66x / 0.71x | 10046 |
| S2 | sparse + xsqmac8 + xqmac8 + xrequant (all) | 340.6 M | 279.9 M | 39.2x / 47.7x | 1.63x / 1.98x | 8978 |

MAC / instruction (real MACs = 1814073344 per image): B0 0.136 / 0.136, B1 0.334 / 0.334, X1 3.095 / 3.086, X2 3.276 / 3.277, S0 0.624 / 0.785, S1 2.149 / 2.31, S2 5.327 / 6.482

### Per-operator instructions per image (M), high-sparsity model

| Operator group | B0 | B1 | X1 | X2 | S0 | S1 | S2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3x3 / 1x1 convs | 11,980.1 | 4,864.1 | 495.4 | 476.7 | 1,731.1 | 206.2 | 189.9 |
| stem conv (7x7, dense) | 1,342.9 | 538.6 | 66.5 | 57.5 | 538.6 | 538.6 | 57.5 |
| CSR compression | 0.0 | 0.0 | 0.0 | 0.0 | 13.0 | 13.0 | 13.0 |
| maxpool | 8.4 | 8.4 | 8.4 | 8.4 | 8.4 | 8.4 | 8.4 |
| residual add + requant | 16.9 | 16.9 | 16.9 | 10.5 | 16.9 | 16.9 | 10.5 |
| avgpool + fc | 3.8 | 1.9 | 0.4 | 0.4 | 1.9 | 1.9 | 0.4 |

### Dynamic instruction mix (1 image, spike -g PC histogram)

| Model | Config | total | load | store | ALU | mul | branch | custom | loads+stores / MAC | custom breakdown |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | B0 | 13,357M | 28.4% | 0.1% | 43.1% | 14.1% | 14.3% | 0.0% | 2.097 |  |
| dense | B1 | 5,435M | 26.5% | 0.2% | 37.7% | 34.7% | 0.9% | 0.0% | 0.801 |  |
| dense | X1 | 593M | 28.9% | 1.5% | 23.6% | 0.7% | 5.2% | 40.1% | 0.099 | xqmac8 22.5M, xqmac8u 215.2M |
| dense | X2 | 560M | 29.5% | 1.4% | 21.9% | 0.3% | 3.8% | 43.0% | 0.096 | xqmac8 22.5M, xqmac8u 215.2M, xrequant 3.2M |
| dense | S0 | 2,894M | 30.3% | 0.5% | 37.5% | 29.4% | 2.3% | 0.0% | 0.490 |  |
| dense | S1 | 853M | 29.5% | 1.3% | 33.3% | 21.6% | 4.2% | 10.0% | 0.145 | xsqmac8 83.4M, xsaccrd 1.7M |
| dense | S2 | 350M | 35.3% | 2.4% | 23.1% | 0.5% | 7.0% | 31.7% | 0.073 | xqmac8 22.5M, xqmac8u 0.1M, xrequant 3.2M, xsqmac8 83.4M, xsaccrd 1.7M |
| sparse | B0 | 13,359M | 28.4% | 0.1% | 43.1% | 14.1% | 14.3% | 0.0% | 2.097 |  |
| sparse | B1 | 5,437M | 26.5% | 0.2% | 37.7% | 34.7% | 0.9% | 0.0% | 0.801 |  |
| sparse | X1 | 594M | 28.9% | 1.5% | 23.8% | 0.7% | 5.2% | 40.0% | 0.099 | xqmac8 22.5M, xqmac8u 215.2M |
| sparse | X2 | 560M | 29.6% | 1.4% | 21.9% | 0.3% | 3.8% | 43.0% | 0.096 | xqmac8 22.5M, xqmac8u 215.2M, xrequant 3.2M |
| sparse | S0 | 2,346M | 30.0% | 0.6% | 37.6% | 29.4% | 2.4% | 0.0% | 0.395 |  |
| sparse | S1 | 799M | 28.7% | 1.3% | 34.6% | 23.1% | 4.2% | 8.1% | 0.132 | xsqmac8 63.2M, xsaccrd 1.7M |
| sparse | S2 | 294M | 34.2% | 2.8% | 24.3% | 0.6% | 7.3% | 30.8% | 0.060 | xqmac8 22.5M, xqmac8u 0.1M, xrequant 3.2M, xsqmac8 63.2M, xsaccrd 1.7M |

### Accumulator-file size sweep (S2, instructions per image, M)

| accumulators | Normal model | High-sparsity model |
|---:|---:|---:|
| 16 | 511.4 | 409.3 |
| 32 | 397.5 | 323.1 |
| 64 | 340.6 | 279.9 |
| 128 | 324.7 | 268.2 |

### S2 dense-dispatch threshold sweep (layers with input density > threshold use dense xqmac8)

| threshold (permille) | Normal model | High-sparsity model |
|---:|---:|---:|
| 500 | 361.1 | 302.9 |
| 600 | 346.2 | 288.3 |
| 700 | 343.2 | 279.9 |
| 800 | 340.4 | 279.9 |
| 900 | 340.6 | 279.9 |
| off | 340.6 | 279.9 |

### Conv-input density observed at runtime in Spike (S2, mean over images)

| conv | normal | high-sparsity |
|---|---:|---:|
| layer1.0.conv1 | 0.767 | 0.571 |
| layer1.0.conv2 | 0.585 | 0.497 |
| layer1.1.conv1 | 0.796 | 0.627 |
| layer1.1.conv2 | 0.466 | 0.302 |
| layer2.0.conv1 | 0.750 | 0.572 |
| layer2.0.downsample.0 | 0.750 | 0.572 |
| layer2.0.conv2 | 0.405 | 0.283 |
| layer2.1.conv1 | 0.501 | 0.322 |
| layer2.1.conv2 | 0.241 | 0.171 |
| layer3.0.conv1 | 0.492 | 0.349 |
| layer3.0.downsample.0 | 0.492 | 0.349 |
| layer3.0.conv2 | 0.354 | 0.257 |
| layer3.1.conv1 | 0.444 | 0.345 |
| layer3.1.conv2 | 0.195 | 0.146 |
| layer4.0.conv1 | 0.436 | 0.332 |
| layer4.0.downsample.0 | 0.436 | 0.332 |
| layer4.0.conv2 | 0.182 | 0.137 |
| layer4.1.conv1 | 0.201 | 0.137 |
| layer4.1.conv2 | 0.185 | 0.113 |

