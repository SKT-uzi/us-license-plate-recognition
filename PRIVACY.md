# Privacy and Release Policy

This repository is designed for code and model demonstration, not for publishing operational license-plate records.

## Included

- Application, API, training, and evaluation code.
- Privacy-scrubbed inference checkpoints.
- Aggregate and anonymized evaluation metrics.
- A public benchmark citation without redistributing its source images or plate identifiers.

## Excluded

- Original training, validation, or test images.
- Customer-provided images, labels, names, account data, or internal identifiers.
- Company names, internal project names, workstation usernames, and absolute local paths.
- Raw license-plate identifiers from benchmark or operational images.
- Runtime uploads, generated crops, logs, caches, and local PaddleOCR downloads.

The `.pt` files were re-saved after replacing training-time path metadata with generic repository-relative values. Tensor contents were compared before and after the metadata rewrite to confirm that model parameters did not change.

## Runtime behavior

The normal HTTP upload endpoints decode and process images in memory. They do not save uploaded images to disk. The optional command-line self-test output mode can save result images when explicitly requested by the operator; those output directories are ignored by Git.

## Before sharing or changing visibility

1. Keep the repository private until the release owner approves it.
2. Do not add datasets, uploaded images, runtime output, caches, or OCR model downloads.
3. Run the repository audit with known organization and customer terms:

   ```bash
   python scripts/privacy_audit.py \
     --deny-term "YOUR_COMPANY_NAME" \
     --deny-term "CUSTOMER_NAME"
   ```

4. Review `git diff --cached` and the complete reachable Git history before pushing.
5. If sensitive data ever reaches Git history, rewrite or recreate the repository; deleting it in a later commit is not sufficient.

This checklist is a technical safeguard, not a substitute for an organization's legal, privacy, and information-security review.
