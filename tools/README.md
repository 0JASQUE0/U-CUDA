# Developer and runtime tools

Tools are auxiliary programs that are not linked into C++ targets. Currently [ocr](ocr/README.md) contains the optional Python recognition server and its pinned environment. Runtime packaging copies the server beside executables; Python packages are installed separately.

## Fixed MNIST pilot

Download the four gzip IDX archives from the [CVDF mirror](https://github.com/cvdfoundation/mnist#download)
into a local directory. The standard-library Python script verifies their
SHA-256 hashes and writes the first 20 training and next five validation
examples of each digit in official training-file order. It takes five
examples per digit from the official test file. IDs retain the original
zero-based IDX indices. Pixel values are scaled by 255 in row-major order.

```powershell
python tools/mnist_subset.py build/mnist build/mnist/mnist-pilot.csv
cmake --build --preset windows-release --target neural_classifier_test
& .\build\windows\bin\Release\neural_classifier_test.exe --mnist-pilot build/mnist/mnist-pilot.csv
```

The pilot uses 784 leaky nodes and routes each pixel to one node. It compares
raw-pixel ridge with the same reservoir graph at `K = 0` and `K = 0.5`.
The graph has fixed in-degree 16, total-degree normalisation, relative
weight spread 2 and coupling `w*tanh(v_j)`. Network seeds are 1, 2 and 3.
Within each seed, the graph topology, routing, initial states, trial resets
and feature nodes are identical. Each ridge is trained independently and
selects its regularisation on the same validation split. The executable
prints test accuracy and the confusion matrix by seed and coupling, and
checks classifier export/import for each reservoir run.

The generated pilot CSV has SHA-256
`87759d81faa0c07ed9b13cf2de9f98f63f1c7976c5258e9cf28b9281bf2660ac`.

| RTX 4090 seed | Raw pixels | K = 0 | K = 0.5 |
|---|---:|---:|---:|
| 1 | 37/50 | 37/50 | 36/50 |
| 2 | 37/50 | 37/50 | 38/50 |
| 3 | 37/50 | 37/50 | 37/50 |

The recurrent mean was 74%, with a range of 72–76% across the three graphs.
Export/import reproduced the predictions and confusion matrices in all six
reservoir runs. This is a fixed small pilot using one test split and one
nonzero coupling strength.

## Full MNIST comparison

Use the same verified archives as the pilot. The final 5,000 official training
examples form the validation split. The official 10,000 test examples stay
in the test split. The generated system and sessions use the pilot graph and
trial settings. The constant image is held on a 0.5-unit input grid to keep
the batch buffer small. The integration step stays at 0.125.

```powershell
python tools/mnist_subset.py build/mnist build/mnist/mnist-full.csv --full
cmake --preset windows -DU_CUDA_BUILD_GUI=OFF
cmake --build --preset windows-release --target neural_classifier_test ucuda_neural_batch
$bin = '.\build\windows\bin\Release'
& "$bin\neural_classifier_test.exe" --write-mnist-sessions build/mnist/full ../mnist-full.csv
foreach ($k in @('0', '0.5')) {
    & "$bin\ucuda_neural_batch.exe" build/mnist/full/system.json "build/mnist/full/mnist-k$k.json" 2048 "build/mnist/full/results-k$k.csv" 1 2 3
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
```

The full CSV SHA-256 is
`85ba3c9326e5c7e3d829f755bae6af4591f5440fb0098c1e8a30dac92622ca2f`.

On RTX 4090, raw pixels and `K = 0` scored 86.27% for every seed. The
`K = 0.5` runs scored 88.06%, 87.84%, and 87.88%, with a mean of 87.93%.
Feature extraction took 86–92 seconds per run. The engine reported peak
device allocations of 128,852,800 bytes with batches of 2,048 images.
The observed Windows process working-set peak was approximately 1.72 GiB
during the control series.

## Sequential columns on MNIST development data

This experiment uses the official training archive only, in IDX order.
Rows 0–49,999 train the readout, rows 50,000–54,999 select its ridge
regularisation, and rows 55,000–59,999 form a development holdout.
The official test set is reserved for a frozen final configuration.

```powershell
python tools/mnist_subset.py build/mnist build/mnist/mnist-dev.csv --development
cmake --build --preset windows-release --target neural_classifier_test ucuda_neural_batch
$bin = '.\build\windows\bin\Release'
& "$bin\neural_classifier_test.exe" --write-mnist-column-sessions build/mnist/columns ../mnist-dev.csv
foreach ($k in @('0', '0.5')) {
    & "$bin\ucuda_neural_batch.exe" build/mnist/columns/system.json "build/mnist/columns/mnist-k$k.json" 2048 "build/mnist/columns/results-k$k.csv" 1
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
```

The first column configuration uses 280 leaky nodes, input gain 4, leak
rate 2 and four feature snapshots after columns 7, 14, 21 and 28. Each
column lasts 0.125 time units, equal to the integration step. For both
couplings, the graph seed is 1 and the 28 row channels each drive ten
nodes. The CSV SHA-256 is
`72780abb8815278f1b0e32b929388e4a3a9ca35308a66a028835eb5bfd044791`.

| Development run, seed 1 | Validation | Holdout |
|---|---:|---:|
| Static recurrent, K = 0.5 | 86.86% | 89.78% |
| Columns, 280 nodes, K = 0 | 77.84% | 80.50% |
| Columns, 280 nodes, K = 0.5 | 92.18% | 93.94% |
| Columns, 560 nodes, K = 0.5 | 93.28% | 95.10% |

The raw-pixel ridge scored 88.28% on this holdout. The 560-node variant
changes population size and feature-node count from 280 to 560. The
four snapshots give 2,240 readout features. For the official comparison,
freeze this architecture and use the full CSV split documented above.

On the official 10,000-image test split, the frozen 560-node architecture
with seed 1 scored 93.92%. Ridge regularisation was selected again using
the full protocol's validation split (95.16% validation, lambda 1e-6).
Feature extraction took 117.09 seconds with batches of 2,048 on RTX 4090.
The engine reported a peak device allocation of 49,842,328 bytes.
