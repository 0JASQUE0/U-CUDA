# Headless Neural classification

`ucuda_neural_batch` takes a system record JSON and a Neural session JSON.
It uses the session's active configuration and loads its dataset CSV. Relative
dataset paths are resolved against the session file directory. It runs one
continuous, fixed-link reservoir at a time using the specified graph seeds.

```powershell
cmake --preset windows -DU_CUDA_BUILD_GUI=OFF
cmake --build --preset windows-release --target ucuda_neural_batch
& .\build\windows\bin\Release\ucuda_neural_batch.exe system.json session.json 2048 results.csv 1 2 3
```

The executable extracts features from separate batches of trials with the
same graph, initial state and input routing. Each image is a distinct trial.
It retains the feature matrix on the host and fits one classifier per seed
on the original training and validation splits. The resulting CSV includes
test accuracy, raw-input ridge accuracy, the confusion matrix, graph hash,
split sizes, extraction time, and peak device allocation reported by the engine.
Failures are recorded by seed and return a nonzero exit code. The output CSV
must be a new file.

For sequential stimuli, set the classification task's `dataset_frames`.
A value of 28 interprets a 784-value row-major vector as 28 successive
columns of 28 values. Each frame is held for one input sample interval,
and the stimulus duration must equal the number of frames times that
interval. The raw-input control still uses the original complete vector.

The [MNIST benchmark procedure](../../tools/README.md#full-mnist-comparison)
describes generation of the input CSV and session files.
