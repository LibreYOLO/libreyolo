# Releasing

Maintainer-only notes for publishing LibreYOLO to PyPI.

## Relevant files

- `MANIFEST.in` excludes weights and other large artifacts from the source distribution.
- `.github/workflows/publish.yml` builds artifacts and publishes to PyPI via Trusted Publishing (OIDC).

## Publishing a new version

1. Bump the version in `pyproject.toml` and treat it as the source of truth.
2. Commit the version bump and push it to `main`.
3. Confirm tests and install smoke checks have passed for the release commit before tagging.
4. Create and publish a GitHub release with tag `vX.Y.Z`:
   `https://github.com/LibreYOLO/libreyolo/releases/new`
5. Open the Actions run and approve the final publish step:
   `https://github.com/LibreYOLO/libreyolo/actions`

## Security

- Publishing approvals are enforced through GitHub Environments:
  `https://github.com/LibreYOLO/libreyolo/settings/environments`
- No PyPI token is stored in GitHub.
- Publishing uses Trusted Publishing (OIDC).
