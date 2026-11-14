# mloda-enterprise

All enterprise plugins for mloda. Source-available: free for development, testing and evaluation; production use needs a license agreement: https://mloda.ai/enterprise

Extras:

- `[ed25519]` Ed25519 manifest signer, and license token signing and verification
- `[otel]` OTel audit log sink
- `[openlineage]` lineage facets extender
- `[anonymizer]` the anonymizer wheel and pyarrow

## Quick start: anonymizer

```bash
pip install "mloda-enterprise[anonymizer]"
```

Platforms: Linux x86_64 and aarch64, macOS x86_64 and arm64, Windows x86_64, Python 3.10 to 3.14 (details on https://pypi.org/project/mloda-anonymizer-binary/).

License key: every run of a binary-backed plugin needs a key issued by mloda, for an evaluation (request one at https://mloda.ai/enterprise) or under a license agreement, read from the first of these that is set: `MLODA_LICENSE_FILE` (path to the key file), `MLODA_LICENSE_KEY` (the token itself). After `exp` a key keeps working for its grace days with a WARNING in the log; renew before. Without a usable key the feature still resolves and the run fails with `LicenseMissingError` or `LicenseInvalidError`; no other plugin takes its place. Development and testing use the simulated binary and test tokens signed with the published test key from `mloda-testing[binary-model]` instead (see the guide linked below).

Key: generate it once with `python -c "import secrets; print(secrets.token_hex(32))"`, store it as a secret and export it as `PII_KEY`, and register it under a credential handle that the `pii_key_handle` option names. A key with leading or trailing whitespace is refused, so strip one read from a file: `Credential(key=Path("pii.key").read_text().strip())`. Keep it stable, a new key gives unrelated pseudonyms. `pii_key_handle` replaces the former `pii_key_env`.

```python
import os

from mloda.user import Credential, DataAccessCollection, Feature, Options, PluginLoader, mloda

PluginLoader.all()

result = mloda.run_all(
    [
        Feature(
            "pseudonymized_email",
            Options(
                group={"pii_key_handle": "pii-key"},
                context={"pseudonymization_algorithm": "hmac_sha256", "in_features": "email"},
            ),
        )
    ],
    compute_frameworks=["PyArrowTable"],
    api_data={"customers": {"email": ["ada@example.com", None]}},
    data_access_collection=DataAccessCollection(credentials={"pii-key": Credential(key=os.environ["PII_KEY"])}),
)
print(result[0])
```

Each value becomes 64 lowercase hex characters; nulls stay null.

Options and key derivation: https://github.com/mloda-ai/mloda-registry/blob/main/docs/guides/feature-group-patterns/29-binary-backed-features.md
