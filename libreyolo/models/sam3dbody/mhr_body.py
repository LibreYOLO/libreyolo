"""MHR (Momentum Human Rig) body model wrapper.

MHR is Meta's parametric human body model, released under Apache 2.0 for both
code and assets. It is the body representation the SAM 3D Body regressor
predicts into, and it is the reason a body-mesh task can exist here at all:
the SMPL family, which the rest of the field standardized on, ships under a
non-commercial license whose model files may not be redistributed, so LibreYOLO
could neither host them nor depend on the ``smplx`` package (whose *code* also
carries that non-commercial license).

Only the TorchScript form of MHR is used. It is a single self-contained file
that needs nothing beyond PyTorch, which avoids the ``pymomentum`` native
dependency that the full MHR package requires and that has no reliable Windows
wheel.

Parameterization, verified against the released model rather than taken from
documentation:

* ``model_params`` is 204 wide: 3 translation, 3 global rotation, 130 body
  pose, then 68 per-bone scales. Rotations are Euler angles in radians, not
  axis-angle, and translation enters the rig in decimeters (hence the factor
  of 10 below).
* ``betas`` is 45 identity blendshape coefficients.
* ``expression`` is 72 facial expression coefficients.
* Outputs are 18439 vertices and a 127-joint skeleton state of
  ``(x, y, z, qx, qy, qz, qw, scale)`` per joint, both in centimeters.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import BinaryIO, Optional, Tuple

import torch

from ._assets import (
    AssetIntegrityError,
    FileIdentity,
    FileSeal,
    PinnedFile,
    atomic_rename_create_only,
    cleanup_private_file,
    ensure_unlinked_directory,
    inspect_pinned_file,
    open_verified_file,
    require_unlinked_directory,
)


logger = logging.getLogger(__name__)


# Public, ungated Apache-2.0 release asset.  The exact release archive and
# member identities were independently checked against GitHub release v1.0.1.
# The archive is not retained after the reviewed member is extracted.
MHR_RELEASE = "v1.0.1"
MHR_SOURCE_REVISION = "4998cec385b1aaa07abdefba71bfba2f83c7db32"
MHR_ASSETS_URL = (
    f"https://github.com/facebookresearch/MHR/releases/download/"
    f"{MHR_RELEASE}/assets.zip"
)
MHR_ARCHIVE = PinnedFile(
    path="assets.zip",
    size=198_943_157,
    sha256="e4f4f205cd87c0fa106577ba1de4fc763e4eb197c924461d2ef7e6944e9d6b94",
)
MHR_ARCHIVE_MEMBER = "assets/mhr_model.pt"
MHR_MODEL_FILE = PinnedFile(
    path="mhr_model.pt",
    size=696_110_248,
    sha256="352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc",
)
MHR_MEMBER_COMPRESSED_SIZE = 26_223_048
MHR_MEMBER_CRC32 = 0x8A4C817C
MHR_LICENSE_MEMBER = "assets/LICENSE.txt"
MHR_LICENSE_URL = (
    f"https://github.com/facebookresearch/MHR/blob/{MHR_SOURCE_REVISION}/LICENSE"
)
MHR_LICENSE_FILE = PinnedFile(
    path="LICENSE",
    size=11_358,
    sha256="cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
)
_DOWNLOAD_TIMEOUT_SECONDS = 60
_COPY_CHUNK_BYTES = 1024 * 1024


class MHRBodyModel(torch.nn.Module):
    """Decode MHR parameters into a posed mesh and skeleton."""

    NUM_BETAS = 45
    NUM_BODY_POSE = 130
    NUM_SCALES = 68
    NUM_EXPRESSION = 72
    NUM_JOINTS = 127
    NUM_VERTICES = 18439
    MODEL_PARAM_DIM = 204

    # The rig optimizes global translation in decimeters; parameters handed in
    # here are meters, so they are scaled on the way in.
    _TRANSLATION_SCALE = 10.0
    # Vertices and joints come back in centimeters.
    _CM_TO_M = 100.0

    def __init__(self, module: torch.jit.ScriptModule):
        super().__init__()
        self.mhr = module
        for param in self.mhr.parameters():
            param.requires_grad = False

    @classmethod
    def from_file(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> "MHRBodyModel":
        """Load the reviewed TorchScript MHR model from a local file."""
        path = Path(path).absolute()
        if not os.path.lexists(path):
            raise FileNotFoundError(
                f"MHR body model not found at {path}. Fetch it with "
                "libreyolo.models.sam3dbody.mhr_body.ensure_mhr_model()."
            )
        require_unlinked_directory(path.parent, label="MHR model directory")
        try:
            # torch.jit.load accepts a file object.  Passing the descriptor that
            # was hashed avoids reopening a mutable pathname after verification.
            with open_verified_file(
                path,
                MHR_MODEL_FILE,
                label="MHR TorchScript model",
            ) as stream:
                module = torch.jit.load(stream, map_location=str(device)).eval()
        except Exception as exc:
            if isinstance(exc, AssetIntegrityError):
                raise
            raise RuntimeError(
                f"could not load the reviewed MHR model at {path}"
            ) from exc
        require_unlinked_directory(path.parent, label="MHR model directory")
        return cls(module).to(device).eval()

    @property
    def faces(self) -> Optional[torch.Tensor]:
        """Mesh topology, when the loaded module exposes it."""
        return getattr(self.mhr, "faces", None)

    def forward(
        self,
        global_orient: torch.Tensor,
        body_pose: torch.Tensor,
        betas: torch.Tensor,
        scales: torch.Tensor,
        transl: Optional[torch.Tensor] = None,
        expression: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pose the body.

        Args:
            global_orient: ``(B, 3)`` root rotation as Euler angles in radians.
            body_pose: ``(B, 130)`` body joint parameters in radians.
            betas: ``(B, 45)`` identity blendshape coefficients.
            scales: ``(B, 68)`` per-bone skeleton scales.
            transl: ``(B, 3)`` root translation in meters. Defaults to zeros,
                which leaves the body at the rig origin so a camera
                translation can be applied afterwards.
            expression: ``(B, 72)`` facial expression coefficients.

        Returns:
            ``(vertices, joints)`` with shapes ``(B, 18439, 3)`` and
            ``(B, 127, 3)``, both in meters in the model's own frame.
        """
        batch = global_orient.shape[0]
        device, dtype = global_orient.device, global_orient.dtype

        if transl is None:
            transl = torch.zeros(batch, 3, device=device, dtype=dtype)
        if expression is None:
            expression = torch.zeros(
                batch, self.NUM_EXPRESSION, device=device, dtype=dtype
            )

        self._check_width("body_pose", body_pose, self.NUM_BODY_POSE)
        self._check_width("betas", betas, self.NUM_BETAS)
        self._check_width("scales", scales, self.NUM_SCALES)

        model_params = torch.cat(
            [transl * self._TRANSLATION_SCALE, global_orient, body_pose, scales],
            dim=1,
        )
        if model_params.shape[1] != self.MODEL_PARAM_DIM:
            raise ValueError(
                f"assembled MHR model parameters are {model_params.shape[1]} wide, "
                f"expected {self.MODEL_PARAM_DIM}"
            )

        vertices, skeleton_state = self.mhr(betas, model_params, expression)
        # Skeleton state packs position, unit quaternion and scale per joint.
        joints = skeleton_state[..., :3]
        return vertices / self._CM_TO_M, joints / self._CM_TO_M

    @staticmethod
    def _check_width(name: str, tensor: torch.Tensor, expected: int) -> None:
        if tensor.shape[-1] != expected:
            raise ValueError(f"{name} must be {expected} wide, got {tensor.shape[-1]}")


def default_mhr_path() -> Path:
    """Location LibreYOLO caches the MHR body model at."""
    root = os.environ.get("LIBREYOLO_MHR_PATH")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "libreyolo" / "mhr" / "mhr_model.pt"


def inspect_mhr_model(path: str | Path) -> FileIdentity:
    """Validate one MHR file and its complete unlinked parent chain."""

    target = Path(path).absolute()
    require_unlinked_directory(target.parent, label="MHR model directory")
    identity = inspect_pinned_file(
        target,
        MHR_MODEL_FILE,
        label="MHR TorchScript model",
    )
    require_unlinked_directory(target.parent, label="MHR model directory")
    return identity


def _mhr_staging_recoveries(target: Path) -> tuple[Path, ...]:
    prefix = f".{target.name}.staging-"
    try:
        entries = list(os.scandir(target.parent))
    except OSError as exc:
        raise AssetIntegrityError(
            f"could not inspect MHR cache directory: {target.parent}"
        ) from exc
    return tuple(
        target.parent / entry.name for entry in entries if entry.name.startswith(prefix)
    )


def ensure_mhr_model(path: str | Path | None = None) -> Path:
    """Return a local MHR model path, downloading the release asset if absent.

    The asset is Apache 2.0 and served from a public GitHub release, so no
    token, registration or license acceptance is involved.
    """
    target = (Path(path) if path is not None else default_mhr_path()).absolute()
    if os.path.lexists(target):
        inspect_mhr_model(target)
        return target

    ensure_unlinked_directory(
        target.parent,
        label="MHR cache directory",
    )
    recoveries = _mhr_staging_recoveries(target)
    if recoveries:
        paths = ", ".join(str(path) for path in recoveries)
        raise AssetIntegrityError(
            "A previous MHR acquisition left private recovery data. Inspect and "
            "remove it before retrying so 700 MB staging files cannot accumulate: "
            f"{paths}"
        )
    logger.info(
        "Downloading the pinned MHR %s body model (Apache 2.0, ~700 MB) from %s",
        MHR_RELEASE,
        MHR_ASSETS_URL,
    )

    with (
        tempfile.TemporaryFile(mode="w+b") as archive,
        tempfile.TemporaryFile(mode="w+b") as extracted,
    ):
        _download_archive(archive)
        _extract_model(archive, extracted)
        staged, staged_seal = _stage_model(extracted, target)

    # Creating the stage changes the parent directory metadata, so capture its
    # stable object identity after staging and immediately before publication.
    parent_seal = require_unlinked_directory(
        target.parent,
        label="MHR cache directory",
    )
    try:
        atomic_rename_create_only(
            staged,
            target,
            expected_source=staged_seal,
            expected_parent=parent_seal,
        )
    except FileExistsError:
        # A concurrent process may have won the create-only race.  Its output
        # is accepted only if it independently matches the same reviewed bytes.
        try:
            inspect_mhr_model(target)
        except BaseException:
            logger.warning(
                "A concurrent MHR publication produced an invalid winner; the "
                "owned staging file is recoverable at %s and the invalid competing "
                "destination remains at %s; neither path was deleted",
                staged,
                target,
            )
            raise
        try:
            cleanup_private_file(
                staged,
                expected_object=staged_seal,
                label="losing MHR staging file",
            )
        except AssetIntegrityError:
            logger.warning(
                "A concurrent MHR download won publication, but its losing staging "
                "file could not be cleaned safely: %s",
                staged,
                exc_info=True,
            )
        return target
    except BaseException:
        logger.warning(
            "MHR publication failed or its outcome was uncertain; recovery may be "
            "at staging path %s or destination path %s; neither path was deleted "
            "because either pathname may have been concurrently replaced",
            staged,
            target,
        )
        raise

    try:
        inspect_mhr_model(target)
    except BaseException:
        logger.warning(
            "MHR post-publication validation failed; recovery may be at staging "
            "path %s or destination path %s; neither path was deleted",
            staged,
            target,
        )
        raise

    logger.info("MHR body model ready at %s (license: %s)", target, MHR_LICENSE_URL)
    return target


def _download_archive(destination: BinaryIO) -> None:
    request = urllib.request.Request(
        MHR_ASSETS_URL,
        headers={"User-Agent": "libreyolo"},
    )
    digest = hashlib.sha256()
    total = 0
    try:
        response = urllib.request.urlopen(
            request,
            timeout=_DOWNLOAD_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        raise RuntimeError(f"could not download pinned MHR archive: {exc}") from exc
    with response:
        declared = response.headers.get("Content-Length")
        if declared is not None:
            try:
                declared_size = int(declared)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "MHR archive returned an invalid Content-Length"
                ) from exc
            if declared_size != MHR_ARCHIVE.size:
                raise RuntimeError(
                    "MHR archive Content-Length does not match the reviewed release "
                    f"({declared_size} != {MHR_ARCHIVE.size})"
                )
        while True:
            chunk = response.read(min(_COPY_CHUNK_BYTES, MHR_ARCHIVE.size - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > MHR_ARCHIVE.size:
                raise RuntimeError("MHR archive exceeded its reviewed byte size")
            digest.update(chunk)
            destination.write(chunk)
    observed = digest.hexdigest()
    if total != MHR_ARCHIVE.size or observed != MHR_ARCHIVE.sha256:
        raise RuntimeError(
            "MHR archive does not match the reviewed release bytes "
            f"(expected {MHR_ARCHIVE.size}/{MHR_ARCHIVE.sha256}, got "
            f"{total}/{observed})"
        )
    destination.flush()
    destination.seek(0)


def _extract_model(archive: BinaryIO, destination: BinaryIO) -> None:
    archive.seek(0)
    try:
        with zipfile.ZipFile(archive) as bundle:
            license_matches = [
                item
                for item in bundle.infolist()
                if item.filename == MHR_LICENSE_MEMBER
            ]
            if len(license_matches) != 1:
                raise RuntimeError(
                    f"MHR archive must contain exactly one {MHR_LICENSE_MEMBER}"
                )
            license_entry = license_matches[0]
            if (
                license_entry.flag_bits & 0x1
                or license_entry.is_dir()
                or license_entry.file_size != MHR_LICENSE_FILE.size
            ):
                raise RuntimeError("MHR archive license metadata does not match v1.0.1")
            license_digest = hashlib.sha256()
            license_size = 0
            with bundle.open(license_entry, "r") as license_stream:
                while True:
                    chunk = license_stream.read(
                        min(
                            _COPY_CHUNK_BYTES,
                            MHR_LICENSE_FILE.size - license_size + 1,
                        )
                    )
                    if not chunk:
                        break
                    license_size += len(chunk)
                    if license_size > MHR_LICENSE_FILE.size:
                        raise RuntimeError(
                            "MHR archive license exceeded its reviewed size"
                        )
                    license_digest.update(chunk)
            if (
                license_size != MHR_LICENSE_FILE.size
                or license_digest.hexdigest() != MHR_LICENSE_FILE.sha256
            ):
                raise RuntimeError("MHR archive license does not match v1.0.1")

            matches = [
                entry
                for entry in bundle.infolist()
                if entry.filename == MHR_ARCHIVE_MEMBER
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"MHR archive must contain exactly one {MHR_ARCHIVE_MEMBER}"
                )
            entry = matches[0]
            if (
                entry.flag_bits & 0x1
                or entry.is_dir()
                or entry.compress_type != zipfile.ZIP_DEFLATED
                or entry.file_size != MHR_MODEL_FILE.size
                or entry.compress_size != MHR_MEMBER_COMPRESSED_SIZE
                or entry.CRC != MHR_MEMBER_CRC32
            ):
                raise RuntimeError("MHR archive member metadata does not match v1.0.1")

            digest = hashlib.sha256()
            total = 0
            with bundle.open(entry, "r") as source:
                while True:
                    chunk = source.read(
                        min(_COPY_CHUNK_BYTES, MHR_MODEL_FILE.size - total + 1)
                    )
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MHR_MODEL_FILE.size:
                        raise RuntimeError(
                            "MHR model member exceeded its reviewed size"
                        )
                    digest.update(chunk)
                    destination.write(chunk)
    except (zipfile.BadZipFile, EOFError) as exc:
        raise RuntimeError("MHR release archive is not a valid ZIP file") from exc
    observed = digest.hexdigest()
    if total != MHR_MODEL_FILE.size or observed != MHR_MODEL_FILE.sha256:
        raise RuntimeError(
            "MHR model member does not match the reviewed bytes "
            f"(expected {MHR_MODEL_FILE.size}/{MHR_MODEL_FILE.sha256}, got "
            f"{total}/{observed})"
        )
    destination.flush()
    destination.seek(0)


def _stage_model(source: BinaryIO, target: Path) -> tuple[Path, FileSeal]:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.staging-",
        suffix=".tmp",
    )
    staged = Path(temporary_name)
    created_identity = os.fstat(descriptor)
    created_seal = FileSeal(
        device=created_identity.st_dev,
        inode=created_identity.st_ino,
        mode=created_identity.st_mode,
        size=created_identity.st_size,
        mtime_ns=created_identity.st_mtime_ns,
        links=getattr(created_identity, "st_nlink", 1),
    )
    digest = hashlib.sha256()
    total = 0
    try:
        with os.fdopen(descriptor, "wb") as destination:
            source.seek(0)
            while True:
                chunk = source.read(
                    min(_COPY_CHUNK_BYTES, MHR_MODEL_FILE.size - total + 1)
                )
                if not chunk:
                    break
                total += len(chunk)
                if total > MHR_MODEL_FILE.size:
                    raise AssetIntegrityError(
                        "verified MHR stream grew while preparing publication"
                    )
                digest.update(chunk)
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
            staged_identity = os.fstat(destination.fileno())
    except BaseException:
        try:
            cleanup_private_file(
                staged,
                expected_object=created_seal,
                label="incomplete MHR staging file",
            )
        except AssetIntegrityError:
            logger.warning(
                "MHR staging failed and its partial file could not be cleaned safely: %s",
                staged,
                exc_info=True,
            )
        raise
    observed = digest.hexdigest()
    if total != MHR_MODEL_FILE.size or observed != MHR_MODEL_FILE.sha256:
        try:
            cleanup_private_file(
                staged,
                expected_object=created_seal,
                label="invalid MHR staging file",
            )
        except AssetIntegrityError:
            logger.warning(
                "MHR staging verification failed and its private file could not be "
                "cleaned safely: %s",
                staged,
                exc_info=True,
            )
        raise AssetIntegrityError("verified MHR stream changed before publication")
    return staged, FileSeal(
        device=staged_identity.st_dev,
        inode=staged_identity.st_ino,
        mode=staged_identity.st_mode,
        size=staged_identity.st_size,
        mtime_ns=staged_identity.st_mtime_ns,
        links=getattr(staged_identity, "st_nlink", 1),
    )


def load_mhr_body_model(
    path: str | Path | None = None,
    device: str | torch.device = "cpu",
    download: bool = True,
) -> MHRBodyModel:
    """Load the MHR body model, fetching it on first use when allowed."""
    target = (Path(path) if path is not None else default_mhr_path()).absolute()
    if not os.path.lexists(target):
        if not download:
            raise FileNotFoundError(
                f"MHR body model not found at {target} and download=False."
            )
        target = ensure_mhr_model(target)
    return MHRBodyModel.from_file(target, device=device)
