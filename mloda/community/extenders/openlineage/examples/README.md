# OpenLineageExtender with Marquez

Start a local Marquez (the OpenLineage reference backend) to view the lineage produced by
`OpenLineageExtender`:

```bash
docker compose -f mloda/community/extenders/openlineage/examples/docker-compose.yaml up
```

Then run a pipeline with the extender pointed at Marquez's HTTP endpoint:

```python
from openlineage.client.client import OpenLineageClient
from mloda.community.extenders.openlineage import OpenLineageExtender
from mloda.user import mloda

client = OpenLineageClient(config={"transport": {"type": "http", "url": "http://localhost:5000"}})

mloda.run_all(
    ["your_feature"],
    function_extender={OpenLineageExtender(client=client)},
)
```

Open the Marquez UI at [http://localhost:3000](http://localhost:3000) to browse the emitted runs
and datasets.
