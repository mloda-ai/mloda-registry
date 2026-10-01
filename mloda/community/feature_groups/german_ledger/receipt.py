"""evidence_receipts: each total's receipt, completed with what the framework knows."""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict
from importlib import metadata
from typing import Any, Optional

from mloda.steward import Extender, ExtenderHook
from mloda.user import mloda

from .policy import AdmissibilityRefused
from .reader import parse_citation
from .sources import RECEIPT_VERSION, InadmissibleTotal

PACKAGE = "mloda-community-german-ledger"
_SUFFIX = "~receipt"


def _package() -> dict[str, Any]:
    try:
        version: str | None = metadata.version(PACKAGE)
    except metadata.PackageNotFoundError:
        version = None
    return {"name": PACKAGE, "version": version}


def _names(groups: Any) -> list[str]:
    return sorted(g.__name__ for g in groups)


def _resolution(diagnosis: Any) -> list[dict[str, Any]]:
    """F1: per feature, which group answered, which others could have, and which declined.

    In mloda more than one group can claim a name, and a subclass is kept over its parent. A
    group that matched but was not chosen was shadowed; that is the audit risk, so it is named.
    """
    if not diagnosis.complete:
        raise ValueError(f"the diagnosis did not resolve: {diagnosis.message}")
    out = []
    for record in diagnosis.records:
        result = record.result
        chosen = set(result.identified)
        out.append(
            {
                "feature": record.feature_name,
                "requested": record.requested,
                "chosen": _names(chosen),
                "also_matched": _names(set(result.criteria_matched) - chosen),
                "abstract": _names(result.abstract_matched),
                "declined": [
                    {"group": g.__name__, "stage": e.stage, "reason": e.reason}
                    for g, e in sorted(result.eliminations.items(), key=lambda kv: kv[0].__name__)
                ],
            }
        )
    return out


def evidence_receipts(result: Any, diagnosis: Any = None) -> list[dict[str, Any]]:
    """One receipt per `<concept>__sources` total in a `mloda.run_all` result.

    The total's own receipt says which policy admitted its rows, which profile, chart and
    catalogue defined it, and who ran it. This adds the framework's side, taken from
    `RunResult` rather than invented: the step that produced the frame, the compute steps of
    the plan, the citations read field by field, and this package's version.

    With a `mloda.diagnose` of the same request it also says why each producer answered
    (`resolution`). A diagnosis whose chosen groups differ from the plan's is refused: it would
    describe some other run. `run_with_receipts` makes both from one set of arguments.

    A receipt written in another receipt version is refused rather than read as this one.
    """
    plan = [
        {"feature_group": s.feature_group_name, "features": sorted(s.feature_names)}
        for s in result.plan
        if s.step_kind == "compute"
    ]
    resolution = None
    if diagnosis is not None:
        resolution = _resolution(diagnosis)
        # Per feature, not per group: a diagnosis of another concept chooses the same groups.
        chosen = sorted({(str(g), str(r["feature"])) for r in resolution for g in r["chosen"]})
        planned = sorted({(str(s["feature_group"]), str(f)) for s in plan for f in s["features"]})
        if chosen != planned:
            raise ValueError(f"the diagnosis does not describe this run: it chose {chosen}, the run planned {planned}")
    package = _package()
    receipts: list[dict[str, Any]] = []
    for step, frame in result.frames():
        for column in frame.column_names:
            if not column.endswith(_SUFFIX):
                continue
            feature = column[: -len(_SUFFIX)]
            origins = frame.column(f"{feature}~origins").to_pylist()
            for raw, cited in zip(frame.column(column).to_pylist(), origins):
                receipt: dict[str, Any] = json.loads(raw)
                if receipt.get("receipt") != RECEIPT_VERSION:
                    raise ValueError(
                        f"{column}: receipt version {receipt.get('receipt')!r}; this reader knows {RECEIPT_VERSION}"
                    )
                receipt.update(
                    feature=feature,
                    produced_by=step.feature_group_name,
                    step_uuid=None if step.step_uuid is None else str(step.step_uuid),
                    plan=plan,
                    citations=[asdict(parse_citation(o)) for o in cited or []],
                    package=package,
                )
                if resolution is not None:
                    receipt["resolution"] = resolution
                receipts.append(receipt)
    return receipts


def refusal_receipt(error: BaseException) -> Optional[dict[str, Any]]:
    """The receipt of a refused run: which refusal stopped it and how its rows were judged.

    A refusal raises, so a run that is refused returns no result to read receipts from. This
    finds the admissibility refusal in the exception chain mloda raised and returns its
    verdict counts, or None when the error is no counted refusal.
    """
    seen: Optional[BaseException] = error
    while seen is not None:
        if isinstance(seen, (AdmissibilityRefused, InadmissibleTotal)) and seen.verdicts is not None:
            return {
                "receipt": RECEIPT_VERSION,
                "refused": type(seen).__name__,
                "message": str(seen),
                "verdicts": dict(seen.verdicts),
            }
        seen = seen.__cause__ or seen.__context__
    return None


class ReceiptContext(Extender):
    """A pass-through extender on calculate_feature, so the step runs under a HookContext.

    mloda builds the HookContext -- run_id, versions and the verified tenant/project/principal --
    only when some extender wraps the step. Without one, a plugin cannot see who ran it. This
    one changes nothing and exists only so the receipt can say.
    """

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


def run_with_receipts(features: Any, **kwargs: Any) -> tuple[Any, list[dict[str, Any]]]:
    """`mloda.run_all` plus its receipts, resolution included, from ONE set of arguments.

    The diagnosis gets every argument it accepts, so it cannot describe a different request
    than the run; evidence_receipts still checks that both chose the same groups per feature.
    A ReceiptContext is added unless the caller already wraps calculate_feature, so the receipt
    can name the run and the verified principal.
    """
    extenders = set(kwargs.get("function_extender") or set())
    if not any(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE in e.wraps() for e in extenders):
        kwargs["function_extender"] = extenders | {ReceiptContext()}
    accepted = set(inspect.signature(mloda.diagnose).parameters)
    diagnosis = mloda.diagnose(features, **{k: v for k, v in kwargs.items() if k in accepted})
    result = mloda.run_all(features, **kwargs)
    return result, evidence_receipts(result, diagnosis)
