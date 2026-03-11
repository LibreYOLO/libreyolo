LibreYOLO full-port checklist

1. Model wrapper
- Add libreyolo/models/<family>/__init__.py
- Add libreyolo/models/<family>/model.py
- Add libreyolo/models/<family>/nn.py
- Add libreyolo/models/<family>/utils.py
- Implement can_load / detect_size / detect_nb_classes
- Implement _init_model / _preprocess / _forward / _postprocess
- Import the family in libreyolo/models/__init__.py
- Optionally expose it from libreyolo/__init__.py

2. Validation
- Pick or add val_preprocessor_class
- Verify _postprocess accepts input_size and letterbox kwargs
- Verify original_size handling is (width, height)
- Run model.val(...) on a tiny dataset

3. Training
- Decide: BaseTrainer path or upstream-wrapper path
- Add libreyolo/models/<family>/trainer.py if using BaseTrainer
- Implement train(...) on the wrapper model
- Ensure outputs include total_loss if using BaseTrainer
- Ensure checkpoint metadata includes nc / names / size / model_family
- Verify a saved checkpoint reloads in a fresh process

4. Export
- Verify model.export(format="torchscript")
- Verify model.export(format="onnx")
- If claiming OpenVINO/NCNN/TensorRT, verify each explicitly
- Confirm _get_preprocess_numpy matches deployment preprocessing

5. Runtime backends
- Review libreyolo/backends/base.py
- Add preprocessing branch if the family is not yolox/yolo9/rfdetr-compatible
- Add output parsing branch if exported tensors have a new layout
- Patch TensorRT-specific logic if needed

6. Tests
- Add unit tests for family detection and size/class detection
- Add export/metadata tests if backend support was added
- Add family to tests/e2e/conftest.py MODEL_CATALOG only after it is stable
- Add tests/e2e/configs/<family>.yaml if config-driven training should cover it
- Run family-relevant e2e tests

7. Repo gotchas
- Registry matching is first-hit-wins
- detect_size_from_filename assumes single-character size codes unless overridden
- _prepare_state_dict() is not used by BaseModel._load_weights() today
- Exported-runtime support is not automatic just because ONNX export succeeds
