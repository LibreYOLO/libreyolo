# Package publishing process

Relevan files:
- MANIFEST.in to exclude weights/artifacts from the package (sdist)
- .github/workflows/publish.yml to run CI + publish to PyPI via OIDC

To publish a new version:
- Bump version in pyproject.toml (source of truth), commit, push to main
- Go to github.com/Libre-YOLO/libreyolo/releases/new and create/publish a release with tag vX.Y.Z (this triggers the workflow)
- Go to the Actions run and approve the publish step at the end (GitHub Environment gate): github.com/Libre-YOLO/libreyolo/actions

Security:
- Publishing approvals are enforced via GitHub Environments: github.com/Libre-YOLO/libreyolo/settings/environments
- No PyPI tokens stored in GitHub; uses Trusted Publishing (OIDC)
