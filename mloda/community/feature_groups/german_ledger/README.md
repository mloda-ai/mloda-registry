# mloda-community-german-ledger

Totals from German ledger exports that carry the ledger lines behind them, or a refusal
that names the reason.

- **Reads:** GDPdU/GoBD dossiers (`index.xml` + data file) and DATEV-Format EXTF 700
  booking batches.
- **Answers:** `revenue__sources` returns the total **and** the cited rows it was summed
  from (`GL.txt@642915ba8de6:6`). A row no admissibility policy cleared never counts.
- **Refuses by name** instead of guessing: an undeclared encoding, a post-cutoff entry into
  a closed period, an old DATEV version, overlapping batches, and so on.

## Quickstart

```python
from datetime import date

from mloda.user import DataAccessCollection, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.german_ledger import AdmissibilityPolicyGroup


class Closing2025(AdmissibilityPolicyGroup):
    """Your host policy: FY2025 closed on 31 Dec, nothing keyed in after 15 Jan 2026."""

    LOCK_DATE = date(2026, 1, 15)
    PERIOD_END = date(2025, 12, 31)


results = mloda.run_all(
    features=["revenue__sources"],
    compute_frameworks={PyArrowTable},
    data_access_collection=DataAccessCollection(folders={"path/to/dossier"}),
)
for table in results:
    print(table.column("revenue__sources~value")[0], table.column("revenue__sources~origins")[0])
```

Three rules keep it predictable:

1. **Import, don't auto-load.** Importing the package registers its feature groups. Do not
   call `PluginLoader.all()` until [mloda#1745](https://github.com/mloda-ai/mloda/issues/1745)
   is fixed: it loads the stock `ReadFileFeature`, which then claims the same names, and the
   request fails with "Multiple feature groups found". For the same reason the package
   declares no entry point yet.
2. **Exactly one policy per process.** Your subclass of `AdmissibilityPolicyGroup` is the
   policy. With none, `revenue__sources` does not resolve. With two, resolution refuses.
   Either way there is no number. Need several checks? List them as rules in that one
   policy (see below).
3. **Write `from mloda.community... import X`, never `import mloda.community...`.** The
   second form rebinds the name `mloda` and hides `mloda.user.mloda`.

## The chain

```
gdpdu_journal            JournalFeatureGroup / DatevJournalFeatureGroup   rows (or legs)
gdpdu_journal__admitted  your AdmissibilityPolicyGroup subclass           rows, cleared + stamped
revenue, receivables     SkrAccountFeatureGroup                           concept at line grain
revenue__sources         SourcesFeatureGroup                              total + citations
```

The policy is a step in the resolution graph, so it shows in `RunResult.plan`. Removing it is a
resolution failure, not an unvetted total.

## Policy rules

`LOCK_DATE`/`PERIOD_END` is shorthand for one `LateEntryCutoff`. A policy that needs several
checks lists them in `RULES` instead (setting both is a `TypeError`):

```
class DatevClosing2025(AdmissibilityPolicyGroup):
    RULES = (PeriodBound(date(2025, 1, 1), date(2025, 12, 31)), Festschreibung())
```

| Rule | Admits | Outside scope | Refuses |
|---|---|---|---|
| `LateEntryCutoff(lock_date, period_end)` | booked by `period_end`, keyed by `lock_date` | booked after `period_end` | keyed late, never keyed, unbooked, no clock column |
| `PeriodBound(start, end)` | booked within `[start, end]` | booked outside it | unbooked, no booking column |
| `Festschreibung()` | DATEV Festschreibung 1 | Festschreibung 0 | no flag, no column |

All rules run in the one policy step. If any rule refuses, the refusal names every rule that
refused. Otherwise each row gets **one** stamp: `admitted:all-of;rules=…` only when every rule
admitted it, and `outside-scope:all-of;…;outside=<rule>` as soon as one did not. Only
admissions are totalled. A single rule stamps under its own name, as before.

**One host, both formats.** The late-entry cutoff needs an `Erfassungsdatum` that DATEV does
not carry, and `Festschreibung` needs a flag that GDPdU does not carry. `ForFormat` scopes a
rule to one format's rows:

```
class MixedClosing2025(AdmissibilityPolicyGroup):
    RULES = (
        ForFormat("gdpdu", LateEntryCutoff(date(2026, 1, 15), date(2025, 12, 31))),
        ForFormat("datev", Festschreibung()),
        PeriodBound(date(2025, 1, 1), date(2025, 12, 31)),
    )
```

A scoped rule does not apply to the other format's rows; it neither admits nor excludes them.
A row that no rule applies to is refused, because nothing vouched for it. A row's format comes
from its citation: a DATEV citation names a leg (`/K`, `/G`), a GDPdU citation never does. In
the stamp, the scoped rule is named `gdpdu.late-entry-cutoff`, `datev.festschreibung`.

## Evidence receipt

Each `<concept>__sources` total carries a third column, `~receipt`, beside `~value` and
`~origins`. The citations name the rows; the receipt names what the total depended on:

| Field | From | Says |
|---|---|---|
| `policies`, `verdicts` | the stamps on every journal row | which policy admitted the rows, and how many rows got each verdict |
| `basis.profile` | `SkrAccountFeatureGroup` | which columns were read as account and amount |
| `basis.chart`, `basis.catalogue` | `SkrAccountFeatureGroup` | the chart (SKR03/SKR04), the catalogue's name, its accounts for this concept, and a fingerprint over the whole catalogue in force |
| `basis.sign` | the journal | `soll-positiv` (DATEV) or `as-declared` (GDPdU) |
| `total`, `lines`, `unpriced` | `SourcesFeatureGroup` | the total and how many lines stand behind it |
| `run` | mloda's `HookContext` | `run_id`, versions, and the `tenant_id`/`project_id`/`principal` a platform set with `mloda.steward.verified_context` |

`run_with_receipts(features, **run_all_kwargs)` runs the request and returns the result with
its receipts. To each receipt it adds what mloda knows:
- from `RunResult.plan` and `frames()`: the producing step and the compute steps of the plan;
- `citations`, each origin read field by field (`file`, `fingerprint`, `record`, `leg`; see
  `parse_citation`);
- this package's version;
- `resolution`, from `mloda.diagnose` on the same arguments: per feature, the group `chosen`,
  the groups that `also_matched` (shadowed, e.g. by a subclass), and those that `declined`, with
  stage and reason. A diagnosis that does not match the run's plan is refused.

```python
from mloda.community.feature_groups.german_ledger import run_with_receipts
from mloda.steward import verified_context

with verified_context(principal="pruefer@kanzlei.example"):
    result, [receipt] = run_with_receipts(["revenue__sources"], compute_frameworks={PyArrowTable}, ...)
receipt["basis"]["catalogue"]   # {"name": "SKR04_2025", "accounts": [[4000, 4499]], "fingerprint": "..."}
receipt["run"]["principal"]     # "pruefer@kanzlei.example"
```

**Why `run_with_receipts` and not a plain `run_all`:** mloda builds a step's `HookContext` only
when an extender wraps the step. `run_with_receipts` adds a pass-through `ReceiptContext`
unless you already wrap `calculate_feature`. After a plain `run_all`, `evidence_receipts(result)`
still works, but `run` is all `None` and there is no `resolution`.

**A refused run is counted too.** A refusal raises, so there is no result to read receipts
from. Instead, `AdmissibilityRefused`, `LateEntryRefused` and `InadmissibleTotal` carry
`verdicts`, the rows counted by kind (`admitted`, `outside-scope`, `refused`,
`unevaluated`, `malformed`, `unstamped`), and say it in the message. `refusal_receipt(error)`
finds that refusal in the exception mloda raised:

```python
try:
    run_with_receipts(["revenue__sources"], ...)
except Exception as error:
    refusal_receipt(error)  # {"refused": "InadmissibleTotal", "verdicts": {"admitted": 7, "outside-scope": 1}, ...}
```

**Mapping to OpenLineage (nothing is emitted).** The receipt is shaped so a lineage consumer
can take it without rework:
- `run.run_id` maps to the OpenLineage run;
- each `plan` step (`PlanStep`) maps to a job;
- the reader's `data_access_identity` (e.g. `datev:<folder>@<sha12>`) maps to the input dataset;
- `policies`/`verdicts`, `basis` and `resolution` map to a custom run facet.

The format is our own and versioned (`RECEIPT_VERSION = 1`). Read fields by name: new fields
may be added within a version. It moves to `declared_attributes` with mloda 0.15
(mloda-registry #887). It is not tamper-proof: anything inside the process can write one.

## Formats

| | GDPdU / GoBD | DATEV EXTF |
|---|---|---|
| Container | folder with `index.xml` | folder of `EXTF_*.csv` batches, one client |
| Versions | descriptor-driven | Versionsnummer 700, Kategorie 21 (510/300 refused) |
| Rows | one per ledger line | two per booking row: `Konto` (K) and `Gegenkonto` (G) |
| Sign | as declared | +Soll / −Haben on the leg's account; Generalumkehr flips it |
| Citation | `GL.txt@<sha12>:<row>` | `EXTF_….csv@<sha12>:<record>/<K\|G>` |
| Personal data | not read beyond the ledger | Stammdaten (category 16) skipped unread; no Buchungstext |

A folder holding both formats is refused, not resolved to one of them.

## Current limits

- **Concepts:** two stated charts, not a licensed DATEV catalogue. `SKR04_2025`: revenue
  4000–4499, receivables 1200–1249. `SKR03_2025`: revenue 8000–8499, receivables 1400–1449.
  The host names its chart once (`SkrAccountFeatureGroup.CHART = "03"`, default `"04"`). A
  DATEV header that states a different SKR is refused (`ChartConflict`). An empty header
  leaves the host's chart as the only word, so a host must state it: under the wrong chart,
  SKR03 Bank 1200 counts as SKR04 receivables.
- **Gross DATEV revenue:** a revenue leg on an automatic account, or with any BU key other
  than 40, includes VAT. The automatic accounts are those Odoo's `l10n_de` templates give a
  taxable default (SKR03 8196, 8300, 8310, 8315, 8400, 8410; SKR04 4186, 4200, 4300, 4310,
  4315, 4400). It is refused by name
  (`GrossRevenueRefused`) because no tax-key table exists yet. Only DATEV journals carry a BU
  key; GDPdU lines are booked as declared.
- **Sign:** each concept reports in its natural direction: revenue credit-positive,
  receivables debit-positive. DATEV legs are signed +Soll / −Haben and say so (`Vorzeichen` =
  `soll-positiv`), so revenue legs are flipped. A GDPdU journal states no convention, and its
  amounts are taken as declared, as a Sachkonten export writes them. An exporter that writes
  debit-positive amounts says so on its profile:
  `LedgerProfile("Konto", "Betrag", sign="soll-positiv")`. A profile that contradicts the
  journal's own `Vorzeichen` is refused.
- **Sachkontenlänge:** the charts state 4-digit accounts. A DATEV Sachkonto with more digits
  is refused by name (`AccountLengthUnsupported`) rather than mapped by a guess. No batch we
  hold uses a longer Sachkontenlänge.
- **Twin fixture:** `tests/fixtures/twin_2025` holds the same four bookings as a GDPdU
  dossier and a DATEV batch. Under one `PeriodBound` policy both give revenue 22 485,06 and
  receivables 7 500,00, each with its own citations (`proofs/proof_twin.py`).
- **DATEV totals:** the late-entry cutoff needs an `Erfassungsdatum`, which DATEV does not
  carry, so it refuses every DATEV journal (fail closed). A DATEV host runs `PeriodBound` +
  `Festschreibung` instead.
- **Frameworks:** PyArrow only.

## Development

```bash
pip install -e ".[dev]"
pytest                                   # from the registry root
GERMAN_LEDGER_CORPUS=/path/to/datev/files pytest   # also run third-party DATEV files
```

Tests: `test_gdpdu_claims.py` (one claim per test), `test_datev.py`, `test_gdpdu_reader.py`,
`test_policy_rules.py` (the composite policy) and `test_readme.py`, which runs the quickstart above. `tests/proofs/` holds scripts that each
register classes process-wide (a shadowing subclass, a second policy, `PluginLoader.all()`),
so the tests run them in a subprocess. Each one prints what it reproduced and can be run on
its own.

The third-party DATEV files are not committed: not all of them are licensed for
redistribution. The ledermann sample in `tests/fixtures/` is MIT (see its NOTICE).
