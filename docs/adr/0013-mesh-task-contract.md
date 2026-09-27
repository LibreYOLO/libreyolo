# ADR 0013: Body-mesh task contract

Status: accepted
Date: 2026-07-28

## Context

Human body mesh recovery estimates a posed 3D body from an image. It is a
well-established research task with a decade of benchmarks, yet no
easy-install computer vision library ships it: the entire field distributes
conda environments, compiled dependencies and registration-gated model files,
and the user-facing story stops at demo scripts. That gap is the opportunity.

The gap exists for a licensing reason, not a technical one. The field
standardized on SMPL and its descendants, and the SMPL family is unusable for
a permissively licensed library:

* The SMPL and SMPL-X model files are non-commercial **and**
  non-redistributable. One archive copy is permitted; mirroring is not.
* The `smplx` PyPI package's **code** carries the same Max Planck
  non-commercial license, not a permissive one. This is the most common
  misconception in the field and it rules out the package as a dependency.
* The SMPL license forbids using the model to train networks for commercial
  deployment, so the restriction reaches into any checkpoint supervised by it.

Meta's MHR (Momentum Human Rig) is the first credible alternative: Apache 2.0
for both code and assets, distributed from a public GitHub release with no
registration, and the output representation of the SAM 3D Body regressor whose
weights are redistributable under the SAM license with passthrough.

## Decision

### The task

Add `mesh` as a task, filename suffix `-mesh`, aliases `body-mesh`, `hmr` and
`human-mesh-recovery`. `smpl` is deliberately not an alias, because nothing
shipped under this task is SMPL and silently accepting the name would imply an
interoperability that does not exist.

### The result payload

`Results.meshes` holds a `Meshes` object whose rows align with
`Results.boxes`, the same arrangement the pose task uses for keypoints. It
carries the parametric core (`global_orient`, `body_pose`, `betas`, `transl`),
the decoded geometry (`vertices`, shared `faces`, `joints3d`, `joints2d`),
`conf`, optional `focal_length`, and an `extras` dict for model-specific
parameters such as skeleton scale, hand pose and facial expression.

The payload is body-model-agnostic by construction. `body_model` names the
parameterization and every count is read back from the tensors, because the
layouts genuinely differ: MHR uses Euler angles rather than axis-angle, a flat
130-wide body-pose vector rather than one triplet per joint (rig joints carry
different degrees of freedom), and 45 identity blendshape coefficients rather
than 10 SMPL betas. Hard-coding SMPL's shapes would have made the first
non-SMPL model a breaking change.

Geometry fields are optional. A model may emit parameters without decoding a
mesh, and that has to stay representable rather than being faked with empty
arrays.

### Coordinates and units

Camera frame of the original image, only. `transl` is metric meters with +z
away from the camera. `vertices` and `joints3d` are metric and already include
`transl`. `joints2d` is in original-image pixels, not crop pixels, so top-down
models must lift their crop camera to the full image before returning.

There is no world or gravity frame in this version, and no field implies one.
Video work (tracking, temporal smoothing, world-frame trajectories) would add
explicitly named `*_world` fields rather than changing the meaning of these.

### Body model

MHR, loaded from its TorchScript form. The TorchScript file is self-contained
and needs nothing but PyTorch, which avoids the `pymomentum` native dependency
that the full MHR package requires and that has no reliable Windows wheel.

Verified against the released asset rather than its documentation:
`model_params` is 204 wide (3 translation, 3 global rotation, 130 body pose,
68 bone scales), `betas` is 45, `expression` is 72, and outputs are 18439
vertices plus a 127-joint skeleton state, all in centimeters. Translation
enters the rig in decimeters. The upstream code comments this block as 127
wide, which is stale; the assertion in the same file says 136.

The asset is fetched from the public upstream release and cached locally rather
than mirrored on the LibreYOLO org: it is freely reachable and 700 MB, so a
second copy would serve no one.

The transport is pinned. LibreYOLO fetches the exact MHR v1.0.1 archive,
checks its reviewed size and SHA-256 while streaming with a hard byte cap, and
extracts only the exact reviewed `assets/mhr_model.pt` ZIP member. The member is
checked again before create-only cache publication. LibreYOLO's standalone MHR
decoder passes the same verified descriptor to `torch.jit.load`; the SAM 3D
Body wrapper instead gives the upstream path-only constructor a pathname and
rehashes it immediately before and after construction. Existing cache files are
never trusted by existence alone.

### Deferred deliberately

* **Export** is gated off with an explicit error, as semantic and point were.
  The runtime metadata contract (which body model, how many betas, whether the
  body-model decoder lives inside the exported graph) must be defined before
  artifacts exist that backends would have to keep reading.
* **Validation** raises a clear error instead of pretending to have data. The
  standard benchmarks (3DPW, EMDB, AGORA) are research-license only and are not
  bundled. The metrics themselves ship as `libreyolo.validation.mesh_metrics`
  (MPJPE, PA-MPJPE, PVE) for use against a dataset the user already holds.
* **Test-time augmentation** is rejected: a horizontal flip swaps left and
  right body parts, so merging flipped mesh parameters is not averaging.
* **Training** is out of scope; mesh training needs its own dataset-licensing
  investigation.

### The first family is wrapped, not ported

SAM 3D Body is the only strong MHR regressor. Its **code** is published under
the SAM License, which is not one of the permissive licenses this project may
derive code from: it carries field-of-use restrictions (no military or warfare
use, no nuclear, espionage, guns or illegal weapons), a no-reverse-engineering
clause, an indemnification obligation, and a term letting Meta amend the
agreement unilaterally. Vendoring or reimplementing it would either put
non-OSI code in an MIT tree or restructure incompatibly-licensed code to
obscure its origin, both of which the licensing policy forbids.

So `LibreSAM3DBody` **wraps** the upstream package rather than porting it. The
adapter is LibreYOLO's own MIT code that calls the upstream public API and
translates its output dict into `Meshes`. The SAM License obligation triggers
on *distributing* SAM Materials; an adapter copies none of their code, and the
upstream package is an optional dependency the user installs themselves. The
practical consequence that matters: a user who never touches the mesh task
never encounters SAM terms at all.

Weights are a separate question from code. The SAM License does permit
redistribution provided the terms are passed through, so the checkpoints are
mirrored on the LibreYOLO org with the license included, behind an automatic
gate. The gate records license acceptance plus self-attested identity and
jurisdiction. It is not manual approval, sanctions screening, or other review
by Meta.

Each mirror is addressed by an immutable Hub commit. During online acquisition,
LibreYOLO requires the reviewed `auto` gate and exact repository tree metadata,
then dry-runs all five reviewed repository files to check the requested commit
and transport-declared size before any runtime payload fetch. Offline
acquisition makes no remote Hub API or network request and checks only the
three runtime objects already present in the local Hub cache. Both paths materialize only
`model.ckpt`, `model_config.yaml`, and `LICENSE` into a revision-keyed private
cache view and hash every local byte. An explicit local checkpoint path must
satisfy the same byte contract.

Online acquisition downloads directly into the private same-filesystem staging
directory, removes Hugging Face's small local metadata tree after strict link
checks, and therefore does not retain a second multi-gigabyte copy in the
shared Hub cache. An incomplete private stage is removed when its filesystem
identity is still the one this process created. A fully verified stage is kept
only when publication fails or an invalid concurrent destination wins; a later
retry refuses to create another stage until the recovery path is inspected and
removed, so repeated failures cannot silently accumulate full checkpoints.

Legacy LibreYOLO caches are the migration exception. Their runtime files are
copied into the current revision-keyed view and retained because those paths
predate this transport and are user-owned. The migration warning names the
three old files and their storage cost so the user can remove them after
validating the new cache.

The upstream SAM 3D Body constructor accepts checkpoint and MHR pathnames only;
it has no descriptor-based loading API. LibreYOLO therefore validates the full
snapshot and MHR file immediately before construction, validates both again
after construction, and fails closed on a persistent change. This is not an
immutable handoff: a same-user process could transiently substitute bytes and
restore them between those checks. Checkpoint, config, license, and MHR paths
must be held on a trusted, quiescent same-user local filesystem while the
constructor runs. Network shares, adversarially writable directories, and a
concurrent process mutating those paths are outside this wrapper's trust model.

Consequences the wrapper accepts: the upstream repository has no packaging
metadata, so it cannot be `pip install`ed and the user must clone it and point
`SAM_3D_BODY_PATH` at it; the upstream estimator moves its batch to the GPU
unconditionally, so there is no CPU path; and the mesh flagship depends on a
third-party layout that could change. Those are the price of not taking their
license into the tree.

## Consequences

The task contract exists independently of any one model, so families can land
incrementally. The MHR decoder is usable on its own to turn parameters into
geometry.

Visualization renders an actual shaded surface, because that is what a body
mesh is expected to look like and a scatter of projected vertices does not
communicate one. The renderer is a small painter's-algorithm rasterizer in
numpy and PIL: back-facing triangles culled, the rest sorted far-to-near and
filled with Lambertian shading. This deliberately avoids `pyrender` and
PyTorch3D, which upstream projects use and which both need a GL context and
install badly on some platforms. Cost is roughly half a second per person at
36874 triangles, only on the `save=True` path. The vertex scatter survives as
a fallback for results that carry projected points but no topology, and
`save_obj()` remains the route to a real 3D file.

The chief cost is that LibreYOLO's body meshes are not SMPL, so the SMPL-shaped
ecosystem (Blender add-ons, retargeting pipelines) needs a conversion step.
Upstream MHR ships SMPL/SMPL-X conversion tools for exactly this. The
alternative, shipping SMPL directly, would have required either a
non-redistributable dependency or a user-supplied-file flow with a
non-commercial cloud over every derived checkpoint. Choosing the permissive
body model keeps the task consistent with the project's licensing posture.
