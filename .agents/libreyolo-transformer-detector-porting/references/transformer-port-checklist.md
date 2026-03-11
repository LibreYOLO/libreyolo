Transformer detector port checklist for LibreYOLO

1. Family detection
- Add a BaseModel subclass in libreyolo/models/<family>/model.py
- Implement can_load() with family-specific transformer keys
- Ensure can_load() does not collide with RF-DETR
- Implement detect_size(); use full checkpoint args/config if needed
- Implement detect_nb_classes(); account for background-inclusive heads

2. Preprocessing and validation
- Decide on direct resize vs letterbox
- Decide on normalization strategy (often ImageNet mean/std)
- Set val_preprocessor_class correctly
- Verify _preprocess() returns original_size as (width, height)
- Verify _postprocess() accepts input_size and letterbox kwargs
- Verify normalized cxcywh -> xyxy pixel conversion is correct

3. Class mapping
- Determine whether labels are contiguous foreground classes
- Handle COCO-91 -> COCO-80 or similar mapping if needed
- Handle background class explicitly if the model predicts num_classes + 1

4. Training
- Choose BaseTrainer vs upstream-wrapper integration
- If using BaseTrainer, ensure forward(..., targets) yields total_loss
- If using upstream-wrapper training, manually reload best checkpoint after train()
- Verify checkpoint reload in a fresh process

5. Export
- Test TorchScript only if the model actually supports it
- Test ONNX with the correct opset
- Override export() if a higher opset is required
- Confirm exported outputs are stable and documented

6. Runtime backends
- Review libreyolo/backends/base.py
- Add family-specific _preprocess() path if needed
- Add family-specific _parse_outputs() path if needed
- Verify runtime metadata preserves model_family

7. Testing
- Add factory/size/class unit tests
- Add validation smoke test
- Add short training test if training is supported
- Add export round-trip tests only for formats that actually work
- Add to tests/e2e/conftest.py MODEL_CATALOG only after stabilization
