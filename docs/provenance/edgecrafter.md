# EdgeCrafter license history and LibreEC provenance

Verified September 13, 2026. This records release history and LibreYOLO's
existing distribution policy; it does not change licenses or download defaults.

## Code and weights have separate terms

| Material | License recorded by LibreYOLO | Scope |
| --- | --- | --- |
| Earlier EdgeCrafter source used by the LibreEC port | Apache-2.0 | Historical attribution, not a license claim for current upstream source |
| Original COCO-only `LibreEC<size>.pt`, `LibreEC<size>-seg.pt`, `LibreEC<size>-pose.pt` | Apache-2.0 | Existing default weight repositories |
| Optional `LibreEC<size>-obj2coco.pt`, `LibreEC<size>-seg-obj2coco.pt`, `LibreEC<size>-pose-obj2coco.pt` | EdgeCrafter License, non-commercial | Objects365 pretraining followed by COCO fine-tuning; separate mirrors |

Here `<size>` is `s`, `m`, `l` or `x`.

The weight licenses are separate from LibreYOLO's MIT license. Commercial use
of the optional mirrors requires a separate upstream license. The full texts
are in [Apache-2.0.txt](../../licenses/Apache-2.0.txt) and
[EdgeCrafter.txt](../../licenses/EdgeCrafter.txt).

## Upstream chronology

- Before the change, the main repository carried Apache-2.0. Its
  [LICENSE at `b9f31f8ce23ac112fac2b9b01bc6a1f25a016d92`](https://github.com/Intellindust-AI-Lab/EdgeCrafter/blob/b9f31f8ce23ac112fac2b9b01bc6a1f25a016d92/LICENSE)
  is the version immediately preceding the replacement.
- August 14, 2026: the separate asset repository published
  [`edgecrafterv1_o365`](https://github.com/capsule2077/edgecrafter/releases/tag/edgecrafterv1_o365),
  including the Objects365-to-COCO checkpoints.
- August 20, 2026: main-project commit
  [`e0656d44b80ecb97ba96bb8a804a741169431e0f`](https://github.com/Intellindust-AI-Lab/EdgeCrafter/commit/e0656d44b80ecb97ba96bb8a804a741169431e0f)
  replaced Apache-2.0 with the custom non-commercial EdgeCrafter License.
  Sections 1, 2, 3 and 5 cover original project code and weights, not only
  Objects365 checkpoints, and require separate licensing for commercial use.
- August 24, 2026: commit
  [`124cb0922ebe4af55c025af448312c2374210815`](https://github.com/Intellindust-AI-Lab/EdgeCrafter/commit/124cb0922ebe4af55c025af448312c2374210815)
  renamed `LICENSE` to `LICENSE.md` with no content changes.

The asset repository still carries
[Apache-2.0 at `b075f34de6947c615557bdb987b0cebebfb7db99`](https://github.com/capsule2077/edgecrafter/blob/b075f34de6947c615557bdb987b0cebebfb7db99/LICENSE).
The Objects365 assets therefore predate the main-project license change and
have conflicting published notices. LibreYOLO's maintainer chose the stricter
EdgeCrafter terms for the optional mirrors. This is a redistribution policy,
not upstream clarification or a claim that earlier Apache grants were revoked.

Apache-2.0 Section 2 grants perpetual, irrevocable copyright permissions,
subject to its conditions; Section 4 sets redistribution obligations.
[EdgeCrafter License Section 10](https://github.com/Intellindust-AI-Lab/EdgeCrafter/blob/124cb0922ebe4af55c025af448312c2374210815/LICENSE.md#10-governing-law-and-governing-terms)
also preserves the terms applicable to a particular release unless otherwise
agreed. The historical grant is the basis for retaining the earlier Apache
attribution; it does not authorize importing later non-commercial source.

## LibreYOLO evidence and limits

- The combined EC detection, segmentation and pose integration was present in
  [LibreYOLO commit `69aaf2b0f89f3a7eb6fd1480ed90dff9d1abba97`](https://github.com/LibreYOLO/libreyolo/commit/69aaf2b0f89f3a7eb6fd1480ed90dff9d1abba97)
  on May 3, 2026 (UTC), during the Apache-2.0 period.
- All 12 default HF repositories were first populated on May 3. As checked on
  September 13, their latest commits were June 27 and each retained Apache-2.0
  and attribution. The [LibreECs card](https://huggingface.co/LibreYOLO/LibreECs/blob/25f25ec5e9621a7d7707290fe3b60420558fc05e/README.md)
  identifies `edgecrafterv1/ecdet_s.pth`; its
  [history](https://huggingface.co/LibreYOLO/LibreECs/commits/25f25ec5e9621a7d7707290fe3b60420558fc05e)
  records the upload, metadata update and ONNX export.
- [LibreYOLO commit `5b52bbc82b786521f205afee12b6a4dc8b013fd3`](https://github.com/LibreYOLO/libreyolo/commit/5b52bbc82b786521f205afee12b6a4dc8b013fd3)
  added the optional variant names and a non-commercial download notice without
  replacing the defaults. All 12 optional mirrors were checked for the complete
  upstream license, a non-commercial banner and a source checksum; see the
  [LibreECs-obj2coco NOTICE](https://huggingface.co/LibreYOLO/LibreECs-obj2coco/blob/780e7323d2f483e7be2314e46b50f970cd588054/NOTICE).
- The original code-attribution entry does not record the exact upstream
  revision used for the port. The license-history commits above are evidence
  of published terms, not recovered source pins. This documentation check is
  not a file-by-file audit of the port, its later changes or third-party
  components, nor a new tensor comparison of the original checkpoints.

See [THIRD_PARTY_NOTICES.txt](../../THIRD_PARTY_NOTICES.txt) for code attribution
and [weights/LICENSE_NOTICE.txt](../../weights/LICENSE_NOTICE.txt) for checkpoint
sources, redistribution terms and the separate dataset terms.
