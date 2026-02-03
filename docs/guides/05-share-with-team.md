# Share a Plugin with Your Team

Share your mloda plugin with team members via a private git repository.

## Prerequisites

Create your plugin using the [mloda-plugin-template](https://github.com/mloda-ai/mloda-plugin-template). See [Create a Plugin Package](04-create-plugin-package.md) for setup instructions.

## Step 1: Push to Private Repo

If you created your repo from the template, it's already on GitHub. Otherwise:

```bash
# Create and push to private repo
gh repo create mycompany/acme-plugins --private --source=. --push
```

## Step 2: Team Members Install

Team members can install directly from the private repo:

```bash
# Install from private repo
pip install git+ssh://git@github.com/mycompany/acme-plugins.git
```

For a specific subdirectory/feature group:

```bash
pip install "acme-scoring @ git+ssh://git@github.com/mycompany/acme-plugins.git#subdirectory=acme/feature_groups/scoring"
```

Then use in code:

```python
from acme.feature_groups.scoring import CustomerScoring
```

## Step 3: Version and Update

Tag releases so team members can pin versions:

```bash
# Tag a release
git tag v1.0.0
git push origin v1.0.0
```

Team members install a specific version:

```bash
pip install git+ssh://git@github.com/mycompany/acme-plugins.git@v1.0.0
```

## Prerequisites

- Team members need git SSH access to the private repo
- For HTTPS, use `git+https://` with appropriate credentials
