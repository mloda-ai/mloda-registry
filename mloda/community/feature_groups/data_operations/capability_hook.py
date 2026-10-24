"""Shared match-time capability hook for the data-operation families (issue #299).

No family uses this hook any more: the families that did moved to ``Implementation.supported``
and ``accepts``. A subclass declares supported subtypes once via ``supported_op_subtypes(secondary)``,
optionally keyed by a secondary axis value.
"""

from __future__ import annotations

from mloda.provider import ComputeFramework
from mloda.user import FeatureName, Options


class SubtypeCapabilityHook:
    # True when supported_op_subtypes depends on a secondary axis. When that axis is
    # unresolved the hook must NOT consult supported_op_subtypes(None): a keyed backend
    # would return the wrong set for an unknown axis value.
    _CAPABILITY_HAS_AXIS: bool = False

    def __init_subclass__(cls, **kwargs: object) -> None:
        # Silently ignoring a pre-#299 override would leave the backend unrestricted and
        # fail later at compute time, so reject the legacy method names loudly here.
        for legacy in ("supported_agg_types", "supported_rank_types"):
            if legacy in cls.__dict__:
                raise TypeError(
                    f"{cls.__name__} defines legacy {legacy}(); rename it to supported_op_subtypes(secondary=None)"
                )
        # Not a legacy registry name: "supported_subtypes" is core's live @final FeatureGroup API.
        # An override collides with it, binds to core's method, and silently leaves this backend
        # unrestricted, so reject the collision explicitly.
        if "supported_subtypes" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} overrides supported_subtypes(), which is core's @final FeatureGroup API; "
                "declare backend capability via supported_op_subtypes(secondary=None) instead"
            )
        if "supported_op_subtypes" in cls.__dict__:
            # A raw classmethod object in __dict__ is not itself callable, so resolve the
            # descriptor via getattr: a frozenset attribute stays non-callable and is caught
            # here, while a legitimate @classmethod override binds to a callable method.
            if not callable(getattr(cls, "supported_op_subtypes")):
                raise TypeError(
                    f"{cls.__name__} binds supported_op_subtypes to a non-callable value; "
                    "define it as a classmethod supported_op_subtypes(secondary=None)"
                )
            # A declared restriction is meaningless without a resolver to check it against.
            if cls._capability_subtype.__func__ is SubtypeCapabilityHook._capability_subtype.__func__:  # type: ignore[attr-defined]
                raise TypeError(
                    f"{cls.__name__} overrides supported_op_subtypes() but does not implement _capability_subtype()"
                )
        # Delegate to super() only after validation so a rejected class never triggers
        # base-class definition hooks. Plugin discovery walks __subclasses__() (a weakref
        # list; see mloda/core/prepare/accessible_plugins.py), so a class whose
        # __init_subclass__ raises is never persistently registered.
        super().__init_subclass__(**kwargs)

    @classmethod
    def supported_op_subtypes(cls, secondary: str | None = None) -> frozenset[str] | None:
        """Subtypes the backend computes natively for the given secondary-axis value; None means unrestricted."""
        return None

    @classmethod
    def _capability_subtype(cls, feature_name: str, options: Options) -> str | None:
        """Resolve the primary discriminator (the agg type); None if unresolvable."""
        raise NotImplementedError(
            f"{cls.__name__} declares supported_op_subtypes() but does not implement _capability_subtype()"
        )

    @classmethod
    def _capability_secondary(cls, feature_name: str, options: Options) -> str | None:
        """Resolve the secondary-axis value; None when the family has no axis."""
        return None

    @classmethod
    def _capability_guard(cls, feature_name: str, options: Options) -> bool:
        """Family-specific up-front rejection (e.g. frame type / time unit); default accept."""
        return True

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        """Reject subtypes the backend cannot compute; unresolvable inputs stay True."""
        name = str(feature_name)
        if not cls._capability_guard(name, options):
            return False
        secondary = cls._capability_secondary(name, options)
        if cls._CAPABILITY_HAS_AXIS and secondary is None:
            return True
        supported = cls.supported_op_subtypes(secondary)
        if supported is None:
            return True
        subtype = cls._capability_subtype(name, options)
        if subtype is None:
            return True
        return subtype in supported
