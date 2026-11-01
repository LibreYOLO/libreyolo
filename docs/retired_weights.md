# Retired weight repositories

LibreYOLO cannot load the checkpoints below, so their Hugging Face repositories
were retired on 2026-11-01. Each model card starts with a "Retired" note that
gives the reason. Found by loading every YOLO9 and RF-DETR repository by name in
the 1.7.0 test campaign; none of them is a regression, and none had downloads.

All 15 are private. To make room in the organization's private storage,
smoke-test and resume-test checkpoints (6.1 GB) were removed from the private
`libreyolo26-research` dataset; its experiment checkpoints and reports are
intact.

| Repository | Why LibreYOLO cannot load it | State |
|---|---|---|
| LibreYOLO9t-cls, LibreYOLO9s-cls | YOLO9 is detection-only: `Task 'classify' is not supported by this model family` | private |
| LibreYOLO9m-cls, LibreYOLO9c-cls, LibreYOLO9t-imagenette-cls | Same | private |
| LibreYOLO9s-pose | YOLO9 is detection-only: `Task 'pose' is not supported` | private |
| LibreRFDETRn-cls, LibreRFDETRs-cls, LibreRFDETRm-cls, LibreRFDETRl-cls, LibreRFDETRn-imagenette-cls | Legacy RF-DETR classification checkpoints; DINOv2 classification loads only `model_family='dinov2'` checkpoints (`LibreDINOv2`) | private |
| LibreRFDETRn-pose, LibreRFDETRs-pose, LibreRFDETRm-pose, LibreRFDETRl-pose | RF-DETR pose ships one size, `LibreRFDETRx-pose`; n, s, m and l were removed in June 2026 (commit `0491505d`), so loading raises `Invalid size` | private |

Kept on purpose:

- `face-rfdetr-nano`: an ONNX face detector for the blur-faces use case and the
  libreyolo-web browser demo, not a LibreYOLO checkpoint.
- `LibreRFDETRn-sem`: a trained semantic-segmentation model that loads from a
  file path; loading it by name does not resolve yet.

To retire another repository: add the card note first, then make it private
(or delete it), and add a row here.
