# Available Layers

## LIBREYOLO8

| Layer | Description |
|-------|-------------|
| `backbone_p1` | First convolution |
| `backbone_p2` | Second convolution |
| `backbone_c2f1` | First C2F block |
| `backbone_p3` | Third convolution |
| `backbone_c2f2_P3` | C2F at P3 (Stride 8) |
| `backbone_p4` | Fourth convolution |
| `backbone_c2f3_P4` | C2F at P4 (Stride 16) |
| `backbone_p5` | Fifth convolution |
| `backbone_c2f4` | Fourth C2F block |
| `backbone_sppf_P5` | SPPF at P5 (Stride 32) |
| `neck_c2f21` | Neck C2F block 1 |
| `neck_c2f11` | Neck C2F block 2 |
| `neck_c2f12` | Neck C2F block 3 |
| `neck_c2f22` | Neck C2F block 4 |
| `head8_conv11` | Head8 box conv |
| `head8_conv21` | Head8 class conv |
| `head16_conv11` | Head16 box conv |
| `head16_conv21` | Head16 class conv |
| `head32_conv11` | Head32 box conv |
| `head32_conv21` | Head32 class conv |

## LIBREYOLO11

| Layer | Description |
|-------|-------------|
| `backbone_p1` | First convolution |
| `backbone_p2` | Second convolution |
| `backbone_c2f1` | First C3k2 block |
| `backbone_p3` | Third convolution |
| `backbone_c2f2_P3` | C3k2 at P3 (Stride 8) |
| `backbone_p4` | Fourth convolution |
| `backbone_c2f3_P4` | C3k2 at P4 (Stride 16) |
| `backbone_p5` | Fifth convolution |
| `backbone_c2f4` | Fourth C3k2 block |
| `backbone_sppf` | SPPF block |
| `backbone_c2psa_P5` | C2PSA at P5 (Stride 32) |
| `neck_c2f21` | Neck C3k2 block 1 |
| `neck_c2f11` | Neck C3k2 block 2 |
| `neck_c2f12` | Neck C3k2 block 3 |
| `neck_c2f22` | Neck C3k2 block 4 |
| `head8_conv11` | Head8 box conv |
| `head8_conv21` | Head8 class conv |
| `head16_conv11` | Head16 box conv |
| `head16_conv21` | Head16 class conv |
| `head32_conv11` | Head32 box conv |
| `head32_conv21` | Head32 class conv |

