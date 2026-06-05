# Releasing

Maintainer-only notes for publishing LibreYOLO to PyPI.

## Relevant files

- `MANIFEST.in` excludes weights and other large artifacts from the source distribution.
- `.github/workflows/publish.yml` builds artifacts and publishes to PyPI via Trusted Publishing (OIDC).

## Publishing a new version

1. Bump the version in `pyproject.toml` and treat it as the source of truth.
2. Commit the version bump and push it to `release`.
3. Confirm tests and install smoke checks have passed for the release commit before tagging.
4. Use the GitHub release page to create and publish tag `vX.Y.Z` targeting `release`:
   `https://github.com/LibreYOLO/libreyolo/releases/new`
5. Open the Actions run and approve the final publish step:
   `https://github.com/LibreYOLO/libreyolo/actions`

The publish workflow rejects release tags that are not reachable from `release`.
The release page creates the tag for you, so there is no separate tag UI step.

## Security

- Publishing approvals are enforced through GitHub Environments:
  `https://github.com/LibreYOLO/libreyolo/settings/environments`
- No PyPI token is stored in GitHub.
- Publishing uses Trusted Publishing (OIDC).
