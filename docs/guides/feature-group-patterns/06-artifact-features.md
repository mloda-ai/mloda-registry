# Pattern 6: Artifact Features (BaseArtifact)

Artifact features save and load fitted state (scalers, models) between runs.

**What**: Features that persist computed state (fitted parameters, trained models, cached results, embeddings, categorical variables).
**When**: Use when computation is expensive or state must be reused (training vs inference).
**Why**: Avoid refitting scalers/models on each run; ensure consistency between train and predict.
**Where**: Fitted sklearn transformers, trained ML models, expensive API responses, lookup tables.
**How**: Pickle, Joblib, Parquet, JSON, or cloud storage (S3, GCS, Azure Blob).

## Key Characteristic

| Method | Behavior |
|--------|----------|
| `artifact()` | Returns `BaseArtifact` subclass |
| `custom_saver()` | Override to save artifact |
| `custom_loader()` | Override to load artifact |

## Complete Example

```python
from typing import Any, Optional, Set, Type
from mloda.provider import FeatureGroup
from mloda.provider import FeatureChainParserMixin, BaseArtifact, FeatureSet
from mloda.user import Feature, Options, FeatureName
import pickle
from pathlib import Path


class ScalerArtifact(BaseArtifact):
    """Store fitted scaler params (mean, std)."""

    @classmethod
    def custom_saver(cls, features: FeatureSet, artifact: Any) -> Optional[str]:
        path = f"/tmp/scaler_{features.name_of_one_feature.name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(artifact, f)
        return path

    @classmethod
    def custom_loader(cls, features: FeatureSet) -> Optional[Any]:
        path = f"/tmp/scaler_{features.name_of_one_feature.name}.pkl"
        if Path(path).exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return None


class StandardScaledFeature(FeatureChainParserMixin, FeatureGroup):
    """Standardize: input__standard_scaled → (x - mean) / std."""

    PREFIX_PATTERN = r"^.+__standard_scaled$"

    @staticmethod
    def artifact() -> Optional[Type[BaseArtifact]]:
        return ScalerArtifact

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        source = feature_name.name.replace("__standard_scaled", "")
        return {Feature.not_typed(source)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        feature_name = features.name_of_one_feature.name
        source = feature_name.replace("__standard_scaled", "")
        col = data[source]

        artifact = ScalerArtifact.custom_loader(features)
        if artifact:
            mean, std = artifact["mean"], artifact["std"]
        else:
            mean, std = col.mean(), col.std()
            ScalerArtifact.custom_saver(features, {"mean": mean, "std": std})

        return (col - mean) / std
```

## Test

```python
import pandas as pd

def test_standard_scaled_feature():
    assert StandardScaledFeature.artifact() == ScalerArtifact

    df = pd.DataFrame({"price": [10, 20, 30]})
    class MockFeatures:
        class name_of_one_feature:
            name = "price__standard_scaled"
    result = StandardScaledFeature.calculate_feature(df, MockFeatures())
    assert abs(result.mean()) < 0.01
```

## Real Implementations

| File | Description |
|------|-------------|
| [sklearn_artifact.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/experimental/sklearn/sklearn_artifact.py) | sklearn model artifact |
| [forecasting_artifact.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/feature_group/experimental/forecasting/forecasting_artifact.py) | Forecasting model artifact |

## Combines With

- **Chained** (Pattern 3): `price__standard_scaled`
- **Multi-output** (Pattern 5): Fitted encoder producing multiple columns
- **Index** (Pattern 7): Time-based model checkpoints
