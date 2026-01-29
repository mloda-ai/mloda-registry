# Feature Naming

How to define what feature names a FeatureGroup provides.

**What**: The mechanism that determines what feature names a FeatureGroup can provide.
**When**: Always - every FeatureGroup needs a naming strategy (default is class name).
**Why**: Users request features by name; mloda must know which FeatureGroup provides each name.
**Where**: Class name, `feature_names_supported()`, or `PREFIX_PATTERN` with FeatureChainParserMixin.

## Naming Mechanisms

| Mechanism | Defines Names | Example |
|-----------|---------------|---------|
| Class name | Single name | `class IsHoliday` → `"IsHoliday"` |
| `feature_names_supported()` | Explicit set | `{"is_holiday", "is_weekend"}` |
| `PREFIX_PATTERN` | Regex pattern | `r"^.+__scaled$"` → `"price__scaled"` |

---

## 1. Class Name (Default)

The class name becomes the feature name automatically.

```python
class IsHoliday(FeatureGroup):
    # Provides feature: "IsHoliday"
    pass
```

---

## 2. Explicit Names

Define a set of supported feature names.

```python
class HolidayFeature(FeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {"is_holiday", "is_weekend", "is_business_day"}
```

---

## 3. Pattern-Based Names (FeatureChainParserMixin)

Define a regex pattern for dynamic feature names.

```python
class ScaledFeature(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r"^.+__scaled$"
    # Provides: "price__scaled", "value__scaled", "anything__scaled"
```

See [Pattern 3: Chained Features](03-chained-features.md).

---

## Name Separators

| Separator | Purpose | Example |
|-----------|---------|---------|
| `__` | Chain input to operation | `price__scaled` |
| `~` | Multi-output columns | `embedding~0`, `embedding~1` |
| `&` | Multiple inputs | `a&b__distance` |

---

## Name Transformation: `set_feature_name()`

Modify the feature name after matching (rarely needed).

```python
def set_feature_name(self, options: Options, feature_name: FeatureName) -> FeatureName:
    # Example: normalize to lowercase
    return FeatureName(feature_name.name.lower())
```

---

## Prefix Convention

By default, `ClassName_anything` also matches via `prefix()`:

```python
class RiskScore(FeatureGroup):
    # Provides: "RiskScore", "RiskScore_v1", "RiskScore_high"
    pass
```

Override for custom prefix:

```python
@classmethod
def prefix(cls) -> str:
    return "risk_"  # Provides: "risk_high", "risk_low"
```
