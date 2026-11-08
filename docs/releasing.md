# Model bundle and release procedure

1. Run the research workflow from the intended source revision and require all steps
   to succeed. Download `ecg-research-bundle`; preserve the source commit and run URL.
2. Inspect class supports, failures, validation selection and external-data scope.
   A successful workflow is an execution check, not proof of useful clinical performance.
3. Retain `comparison.json`, `manifest.json`, `requirements-lock.txt`, weights,
   explorer examples and external reports together. Keep datasets out of Git.
4. In the recorded environment, verify the bundle before serving:

```bash
python -c "from src.models.artifacts import verify_bundle; verify_bundle('artifacts/research')"
```

5. Run the test suite and container smoke workflow. Version the source and bundle
   together. Attach the complete bundle and figures to a reviewed GitHub release
   when ready to publish; this project does not automatically publish releases.

The scientific lock records core numerical dependencies, not every transitive
package or operating-system detail. For a deployment, retain the full environment
and image digest too. A manifest is an integrity contract, not a signature or
authorization to deserialize untrusted pickle files.

External reports identify the manifest used at evaluation time. They may be added
alongside the original immutable model bundle without changing the model manifest.
Their hashes are not covered by that original manifest; archive the full workflow
artifact and run provenance when distributing them.
