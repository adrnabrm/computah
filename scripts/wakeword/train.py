from livekit.wakeword import (
    WakeWordConfig,
    load_config,
    run_generate,
    run_augment,
    run_extraction,
    run_train,
    run_export,
    run_eval,
)

# Load from YAML or construct directly
config = load_config("configs/prod.yaml")

# Run individual stages
run_generate(config)     # TTS synthesis + adversarial negatives
run_augment(config)      # Add noise, reverb, pitch shifts
run_extraction(config)   # Extract mel spectrograms + speech embeddings → .npy
run_train(config)        # 3-phase adaptive training
onnx_path = run_export(config)       # Export to ONNX

# Evaluate the exported model
results = run_eval(config, onnx_path)
print(f"AUT={results['aut']:.4f}  FPPH={results['fpph']:.2f}  Recall={results['recall']:.1%}")