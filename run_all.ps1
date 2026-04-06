Write-Host "=== TRAIN MODEL ==="
python src/train_full_detector.py

Write-Host ""
Write-Host "=== FIND BEST THRESHOLD ==="
python src/find_best_threshold.py

Write-Host ""
Write-Host "=== EVALUATE SOLUTION ==="
python src/evaluate_solution.py

Write-Host ""
Write-Host "=== BENCHMARK LATENCY ==="
python src/benchmark_latency.py

Write-Host ""
Write-Host "=== SAMPLE PREDICTION ==="
python src/predict_detector.py

Write-Host ""
Write-Host "=== DONE ==="
