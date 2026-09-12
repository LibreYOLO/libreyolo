# The LibreYOLO Sponsorship Program

Version 1, September 2026.

## Abstract

LibreYOLO is an MIT-licensed computer vision library. Its goal is to make
machine learning models more accessible, with a particular focus on computer
vision.

After approximately nine months of building in public in an economically lean
fashion, most of the low-hanging fruit has been taken, while some real
advancements that could revolutionize the computer vision field have been
avoided because of their cost. Our ambition is to address these, thanks to the
LibreYOLO Sponsorship Program.

This document defines how the project accepts money, what it does with it,
and what sponsors receive.

## 1. Purpose

The program exists to fund work that LibreYOLO cannot do without money. The
list is detailed in section 2.

The program is not a way to pay the maintainer. Funds go to compute,
benchmarks and infrastructure, as listed in section 2.

## 2. What needs funding

1. **LibreYOLO26.** The first LibreYOLO model built to be competitive,
   trained from scratch and released under MIT in 2026. The bar is
   YOLOv8-class on two measures: accuracy, as COCO mAP at comparable size and
   speed, and generalization, as the RF100-VL score across 100 real-world
   datasets. **LibreYOLO27** follows in 2027 with a higher bar.
2. **Models retrained from scratch.** Several families in the library have
   permissively licensed code but weights that users cannot ship freely,
   while the datasets they need are open. Retraining them on open data gives
   the community MIT weights. This work starts with YOLO-NAS.
3. **Training and inference support for vision-language models.** LibreYOLO
   aims to be the reference library for training VLMs, with the same train,
   val and predict API as its detectors and the same dataset formats.
4. **Benchmarks for visionanalysis.org.** Vision Analysis has the ambition to
   become the go-to reference for benchmarks and the model registry for computer
   vision. We want to benchmark every model on all relevant hardware and
   cover every computer vision task. Ultimately we want to become the
   "Artificial Analysis of computer vision".

Other needs may arise as the library advances.

## 3. Sponsors

Two words are used below. Individuals who contribute are supporters and are
listed by name. Companies that contribute are sponsors and are listed by
logo.

| Who | Contribution | Recognition |
| --- | --- | --- |
| Individuals | any amount | Name in the supporters list below. |
| Companies | $100 per month | Logo and link in the README and on libreyolo.com. |
| Companies | $500 per month | The above, plus named in every release post and in the model card of every weight released while the sponsorship is active, as "Training compute sponsored by". |
| Companies | custom | Ask. |
| Hardware manufacturers | a device | LibreYOLO makes the export formats that device supports work on it, or adds export support for that specific device. Logo and link in the README and on libreyolo.com. The device stays with the project and the benchmark results are published as measured. |

Contributions go through [GitHub Sponsors](https://github.com/sponsors/EHxuban11),
monthly or one-time. Companies that need an invoice or a bank transfer
contribute through [Open Collective](https://opencollective.com/libreyolo),
where every transaction is public. GPU providers may contribute compute
instead of money; it counts at market value.

## 4. Limits

- **No influence on the roadmap.** Sponsorship does not create any ability
  to direct what LibreYOLO builds, which issues get attention, or which pull
  requests are merged.
- **No endorsement.** Being listed does not mean LibreYOLO endorses a
  sponsor, and sponsoring does not mean the sponsor endorses LibreYOLO's
  decisions.

## 5. Transparency

All sponsorship money is loaded as credit into the project's Vast.ai
account. Nothing is tracked per experiment. Once a year this document gets
one line: how much came in and how much was loaded onto Vast.ai, with the
Vast.ai billing history as the record. The sponsor list on GitHub Sponsors
is public.

## 6. Changes

Changes to this program are made by pull request and announced in the next
release notes.

## Sponsors (companies)

Companies and hardware manufacturers that contribute, listed by logo.

None yet.

## Supporters (individuals)

People who contribute any amount, listed by name.

None yet.
