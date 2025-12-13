# Dev Containers Setup

This setup allows you to develop and test your code inside a Docker container. It uses Docker Compose to manage multiple containers and ensures that your development environment is consistent across different machines. The containers are configured to open in the `/workspace` directory, where your project files are located.

## Available Containers

- **reg-agent-1**: Port 8010
- **reg-agent-2**: Port 8011
- **reg-agent-3**: Port 8012
- **reg-agent-4**: Port 8013
- **reg-agent-5**: Port 8014
- **reg-agent-6**: Port 8015

## How to Attach to a Dev Container via vscode

1. Press `Ctrl+Shift+P` to open the command palette.
2. Select `Remote-Containers: Attach to Running Container...`.
3. Choose the desired container from the list.

## How to Attach to a Dev Container via terminal

- **docker exec -it reg-agent-1 /bin/bash**
- **docker exec -it reg-agent-2 /bin/bash**
- **docker exec -it reg-agent-3 /bin/bash**
- **docker exec -it reg-agent-4 /bin/bash**
- **docker exec -it reg-agent-5 /bin/bash**
- **docker exec -it reg-agent-6 /bin/bash**

## Building and Starting Containers

### Build and Start ALL Containers (6 containers)

```sh
cd .devcontainer && docker compose --profile build-only build && docker compose up -d
```

Or use the shorthand (builds base image automatically):
```sh
cd .devcontainer && docker compose up --build -d
```

### Build and Start a SINGLE Container

```sh
# For reg-agent-1 (builds base image if needed)
cd .devcontainer && docker compose up --build -d reg-agent-1

# For reg-agent-2
cd .devcontainer && docker compose up --build -d reg-agent-2

# For reg-agent-3
cd .devcontainer && docker compose up --build -d reg-agent-3

# For reg-agent-4
cd .devcontainer && docker compose up --build -d reg-agent-4

# For reg-agent-5
cd .devcontainer && docker compose up --build -d reg-agent-5

# For reg-agent-6
cd .devcontainer && docker compose up --build -d reg-agent-6
```

### Build and Start Multiple Specific Containers

```sh
# Example: Start only reg-agent-1 and reg-agent-3
cd .devcontainer && docker compose up --build -d reg-agent-1 reg-agent-3
```

### Rebuild Base Image Only

```sh
# When you need to update the base image
cd .devcontainer && docker compose build mloda-registry-base
```

## Stopping Containers

### Stop ALL Containers and Remove Volumes

```sh
docker compose down -v
```

### Stop ALL Containers Without Removing Volumes

```sh
docker compose down
```

### Stop a SINGLE Container

```sh
# For reg-agent-1
docker compose stop reg-agent-1

# For reg-agent-2
docker compose stop reg-agent-2

# For reg-agent-3
docker compose stop reg-agent-3

# For reg-agent-4
docker compose stop reg-agent-4

# For reg-agent-5
docker compose stop reg-agent-5

# For reg-agent-6
docker compose stop reg-agent-6
```

### Remove a SINGLE Container (keeping volume)

```sh
# Example for reg-agent-1
docker compose rm -f reg-agent-1
```

### Remove a SINGLE Container AND its Volume

```sh
# Example for reg-agent-1
docker compose rm -f reg-agent-1
docker volume rm .devcontainer_reg_agent_1_data
```

## Managing Docker Volumes

### List All Project Volumes

```sh
docker volume ls | grep devcontainer
```

### Prune All Unused Docker Volumes

```sh
docker volume prune -f
```
