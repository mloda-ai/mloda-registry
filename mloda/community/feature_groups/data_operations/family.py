"""How a data-operation family describes itself to the catalog and the lints.

The catalog and the drift lints read a family's own answers instead of keeping their
own copy of its vocabulary, so a family is described in exactly one place. The probe
helpers are module-level because several families need the same partition context to
reach match time at all.
"""

from __future__ import annotations

from typing import ClassVar

from mloda.core.abstract_plugins.components.options import Options


def partition_probe_options() -> Options:
    """Options carrying the partition_by context the group-by families require to match."""
    return Options(context={"partition_by": ["region"]})


def partition_order_probe_options() -> Options:
    """Options carrying partition_by and order_by for the ordered-partition families."""
    return Options(context={"partition_by": ["region"], "order_by": "ts"})


class DataOperationFamily:
    """How a data-operation family describes itself to the catalog and the lints."""

    #: Core only ever reads this via getattr and never declares it, so annotating it here is safe.
    PREFIX_PATTERN: ClassVar[str]

    #: Catalog name of the family. Declared in each family's own body, never inherited.
    FAMILY_NAME: ClassVar[str]

    #: What one entry of catalog_subtypes() is called in the generated documentation.
    SUBTYPE_LABEL: ClassVar[str] = "op"

    @classmethod
    def catalog_subtypes(cls) -> tuple[str, ...] | None:
        """The subtype universe in documentation order, or None for a family without a subtype axis."""
        return None

    @classmethod
    def catalog_probe(cls, subtype: str) -> tuple[str, Options]:
        """A feature name and Options that carry *subtype* all the way to match time."""
        raise NotImplementedError(f"{cls.__name__} declares subtypes but no catalog_probe()")

    @classmethod
    def example_feature_names(cls) -> tuple[str, ...]:
        """The family's valid feature names, built from its own live vocabulary."""
        raise NotImplementedError(f"{cls.__name__} declares no example_feature_names()")

    @classmethod
    def matching_patterns(cls) -> tuple[str, ...]:
        """Every name pattern the family routes on; a family matching on more than one lists them all."""
        return (cls.PREFIX_PATTERN,)
