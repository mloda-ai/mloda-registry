# OtelExtender demo stack

An OTel Collector, Grafana Tempo and Grafana, wired together so spans emitted by
`OtelExtender` are visible as traces.

```bash
docker compose up
```

Then point the OTLP SDK exporter at the collector and attach `OtelExtender()` to a run:

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from mloda.community.extenders.otel import OtelExtender
from mloda.user import mloda

provider = TracerProvider(resource=Resource.create({"service.name": "mloda-demo"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)))
trace.set_tracer_provider(provider)

mloda.run_all(["your_feature"], function_extender={OtelExtender()})
provider.force_flush()
```

Open Grafana at `http://localhost:3000`, Explore, Tempo, and search for the
`mloda.calculate` / `mloda.validate.input` / `mloda.validate.output` spans of the run.

Tear down with `docker compose down -v`.
