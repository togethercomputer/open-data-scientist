# 🐳 Python Code Interpreter Service

A containerized Python code execution service that provides an API for running Python code in isolated sessions. This service is designed for local development and experimentation.

## ⚠️ Local Development Tool

**This is intended for local development only** - not for production deployment. The service allows arbitrary code execution and is designed to run in a disposable Docker container.

## 🚀 Quick Start

1. **Start the service:**
   ```bash
   cd interpreter
   docker-compose up --build -d
   ```

2. **Verify it's running:**
   ```bash
   curl http://localhost:8123/health
   ```

3. **Stop the service:**
   ```bash
   docker-compose down
   ```

## 🔒 Security Model

### ✅ Container Isolation
- All code runs inside a Docker container
- Container is **disposable** - can be rebuilt if corrupted
- No access to host system beyond mounted directories

### 🚨 Host Directory Access Warning

The Docker container has **read-write access** to specific host directories:

```yaml
volumes:
  - ../eval/kaggle_data/spooky:/app/spooky:rw
  - ../eval/kaggle_data/jigsaw:/app/jigsaw:rw
```

**⚠️ This means executed code can:**
- **Modify or delete** files in `../eval/kaggle_data/spooky`
- **Modify or delete** files in `../eval/kaggle_data/jigsaw`
- **Create new files** in these directories

**✅ Code CANNOT access:**
- Your home directory
- Other projects
- System files
- Any directories outside the mounted volumes

### 🔧 Recommended: Read-Only Mounts

For safer operation, consider changing to read-only mounts in `docker-compose.yml`:

```yaml
volumes:
  - ../eval/kaggle_data/spooky:/app/spooky:ro  # read-only
  - ../eval/kaggle_data/jigsaw:/app/jigsaw:ro  # read-only
```

## ⚠️ Session Isolation Limitations

While the service provides basic session isolation, it has important limitations:

### ✅ What IS Isolated:
- **User variables**: `x = 1` in one session won't affect another session
- **Session state**: Each session maintains its own execution context

### ❌ What is NOT Isolated:
- **Module modifications**: Changes to imported libraries affect ALL sessions
- **Global state changes**: Modifications to `sys.path`, `os.environ`, etc. are shared
- **Library monkey-patching**: Modifying `json.dumps`, `numpy` settings, etc. corrupts other sessions

### Examples of Problematic Code:
```python
# These operations will affect ALL sessions:
import json
json.dumps = custom_function  # ❌ Breaks all sessions

import sys
sys.path.append('/custom/path')  # ❌ Affects all sessions

import os
os.environ['KEY'] = 'value'  # ❌ Global environment change
```

### Safe for Single-User Local Development:
- **Data analysis workflows**: Reading CSV/JSON files, pandas operations
- **Machine learning**: Training models, feature engineering
- **Visualization**: Creating plots with matplotlib, seaborn
- **Standard data science**: EDA, data cleaning, hypothesis testing

## 📁 File Structure

- `main.py` - FastAPI application with endpoints
- `code_executor.py` - Core code execution logic
- `session_manager.py` - Session handling and isolation
- `download_data.py` - Data download from HuggingFace
- `Dockerfile` - Container configuration
- `docker-compose.yml` - Service orchestration
- `requirements.txt` - Python dependencies

## 🧹 Cleanup

To completely remove the service and data:

```bash
# Stop and remove containers
docker-compose down

# Remove downloaded data
rm -rf downloaded_data/

# Remove custom uploaded data
rm -rf custom_data/

# Remove Docker images (optional)
docker image prune -f
``` 