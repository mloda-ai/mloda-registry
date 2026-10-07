# mloda-enterprise

All enterprise plugins for mloda. Source-available, license required: https://mloda.ai/enterprise

Extras:

- `[ed25519]` Ed25519 manifest signer
- `[otel]` OTel audit log sink
- `[openlineage]` lineage facets extender
- `[anonymizer]` the anonymizer wheel and pyarrow

## Quick start: anonymizer

```bash
pip install "mloda-enterprise[anonymizer]"
```

Platforms: Linux x86_64 and aarch64, macOS x86_64 and arm64, Windows x86_64, Python 3.10+ (details on https://pypi.org/project/mloda-anonymizer-binary/).

License: the binary reads `MLODA_LICENSE_FILE` (path to the license file) first, then `MLODA_LICENSE_KEY` (the token itself). After `exp` a license keeps working for its grace days with a WARNING in the log; renew before.

Key: generate it once with `python -c "import secrets; print(secrets.token_hex(32))"`, store it as a secret and export it as `PII_KEY` (the env var `pii_key_env` names). Keep it stable, a new key gives unrelated pseudonyms.

```python
from mloda.user import Feature, Options, PluginLoader, mloda

PluginLoader.all()

result = mloda.run_all(
    [
        Feature(
            "pseudonymized_email",
            Options(
                context={
                    "pseudonymization_algorithm": "hmac_sha256",
                    "pii_key_env": "PII_KEY",
                    "in_features": "email",
                }
            ),
        )
    ],
    compute_frameworks=["PyArrowTable"],
    api_data={"customers": {"email": ["ada@example.com", None]}},
)
print(result[0])
```

Each value becomes 64 lowercase hex characters; nulls stay null.

Options and key derivation: https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/29-binary-backed-features.md
