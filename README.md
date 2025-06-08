# 🤖 ReAct Data Science Agent

An AI-powered data analysis assistant that follows the ReAct (Reasoning + Acting) framework to perform comprehensive data science tasks. The agent can execute Python code either locally via Docker or in the cloud using Together's Code Interpreter.

## 🚀 Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager
- Together AI API key (get one at [together.ai](https://together.ai))
- Docker and Docker Compose (for local execution mode)

### Install from Source

1. **Clone the repository:**
   ```bash
   cd open-data-scientist
   ```

2. **Install the package:**
   ```bash
   # Install uv (faster alternative to pip)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Create and activate virtual environment
   uv venv --python=3.12
   source .venv/bin/activate
   uv pip install -e .
   ```

3. **Set up your API key:**
   ```bash
   export TOGETHER_API_KEY="your-api-key-here"
   ```
   
   Or add it to your shell profile (`.bashrc`, `.zshrc`, etc.):
   ```bash
   echo 'export TOGETHER_API_KEY="your-api-key-here"' >> ~/.zshrc
   source ~/.zshrc
   ```

## 🎯 Execution Modes

The ReAct agent supports two execution modes for running Python code:

| Feature | TCI (Together Code Interpreter) | Docker/Internal |
|---------|--------------------------------|-----------------|
| **Execution Location** | ☁️ Cloud-based (Together AI) | 🏠 Local Docker container |
| **Setup Required** | API key only | Docker + docker-compose |
| **File Handling** | ☁️ Files uploaded to cloud | 🏠 Files stay local |
| **Session Persistence** | ✅ Managed by Together | ✅ Local session management |
| **Session Isolation** | ✅ Independent isolated sessions | ⚠️ Limited isolation (see below) |
| **Concurrent Usage** | ✅ Multiple users/processes safely | ⚠️ File conflicts possible |
| **Dependencies** | Pre-installed environment | Custom Docker environment |
| **Plot Saving** | ✅ Can save created plots to disk | ❌ Plots not saved to disk |

## ⚠️ Important Privacy Warning

**TCI Mode**: Using TCI will upload your files to Together AI's cloud servers. Only use this mode if you're comfortable with your data being processed in the cloud.

**Docker Mode**: All code execution and file processing happens locally in your Docker container.

## ⚠️ Docker Mode Session Isolation Limitations

**Important**: While Docker mode provides basic session isolation for variables, it has significant limitations:

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

### Docker Mode is OK For:
- **Data analysis workflows**: Reading CSV/JSON files, pandas operations, statistical analysis
- **Machine learning**: Training models, feature engineering, model evaluation
- **Visualization**: Creating plots with matplotlib, seaborn, plotly
- **Standard data science**: EDA, data cleaning, hypothesis testing
- **Single-user development** and **testing environments**

### When to Use TCI Mode Instead:
- **Multi-user environments** where sessions must be completely isolated
- **Production applications** with concurrent users
- **Workflows that modify global state** (if unavoidable)

## 🛠️ Usage

### 🖥️ Command Line Interface

The easiest way to get started is using the command line interface:

```bash
# Basic usage with local Docker execution
open-data-scientist

# Use cloud execution with TCI
open-data-scientist --executor tci

# Specify a custom model and more iterations
open-data-scientist --model "deepseek-ai/DeepSeek-V3" --iterations 15

# Use specific data directory
open-data-scientist --data-dir /path/to/your/data

# Combine options
open-data-scientist --executor tci --model "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo" --iterations 20 --data-dir ./my_data
```

#### CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--model` | `-m` | Language model to use | `deepseek-ai/DeepSeek-V3` |
| `--iterations` | `-i` | Maximum reasoning iterations | `20` |
| `--executor` | `-e` | Execution mode: `tci` or `internal` | `internal` |
| `--data-dir` | `-d` | Data directory to upload | Current directory (with confirmation) |
| `--session-id` | `-s` | Reuse existing session ID | Auto-generated |
| `--help` | `-h` | Show help message | - |

#### Smart Data Directory Handling

- **No directory specified**: The CLI will show files in your current directory and ask if you want to upload them
- **Directory specified**: Validates the path exists and uploads all supported file types
- **Interactive confirmation**: Always asks before uploading files to ensure you know what's being shared

#### Examples

```bash
# Quick start - analyze data in current folder
open-data-scientist

# Use cloud execution for better performance
open-data-scientist --executor tci

# Analyze specific dataset with custom settings
open-data-scientist --data-dir ./sales_data --iterations 25 --model "deepseek-ai/DeepSeek-V3"

# Continue previous session
open-data-scientist --session-id "your-session-id"
```

### 🐍 Python API

For programmatic usage, you can also use the Python API directly:

### Basic Agent Initialization

```python
from open_data_scientist.codeagent import ReActDataScienceAgent

# Cloud execution with TCI
agent = ReActDataScienceAgent(
    executor="tci",
    data_dir="path/to/your/data",  # Optional: auto-upload files
    max_iterations=10
)

# Local execution with Docker
agent = ReActDataScienceAgent(
    executor="internal", 
    data_dir="path/to/your/data",  # Optional: auto-upload files
    max_iterations=10
)
```

### TCI Mode Setup

```python
# Simple TCI usage
agent = ReActDataScienceAgent(executor="tci")
result = agent.run("Analyze the iris dataset and create a scatter plot")

# TCI with data directory (uploads files to cloud)
agent = ReActDataScienceAgent(
    executor="tci",
    data_dir="/path/to/data",
    max_iterations=15
)
result = agent.run("Explore the uploaded CSV files and create summary statistics")
```

### Docker Mode Setup

1. **Start the interpreter service:**
   ```bash
   cd interpreter
   docker-compose up -d
   ```

2. **Use the agent:**
   ```python
   agent = ReActDataScienceAgent(
       executor="internal",
       data_dir="/path/to/data",  # Files uploaded to local container
       max_iterations=10
   )
   result = agent.run("Load the data and perform exploratory data analysis")
   ```
### Advanced Configuration

```python
agent = ReActDataScienceAgent(
    session_id="my-custom-session",  # Reuse existing session
    model="deepseek-ai/DeepSeek-V3",  # Custom LLM model
    max_iterations=20,
    executor="tci",  # or "internal"
    data_dir="/path/to/data"
)

# Run analysis
result = agent.run("""
    Please analyze the uploaded data:
    1. Load all CSV files and show their structure
    2. Identify missing values and outliers
    3. Create visualizations for key patterns
    4. Provide insights and recommendations
""")
```

## 🐳 Docker Mode Detailed Setup

1. **Navigate to interpreter directory:**
   ```bash
   cd interpreter
   ```

2. **Build and start services:**
   ```bash
   docker-compose up --build -d
   ```

3. **Verify service is running:**
   ```bash
   curl http://localhost:8000/health
   ```

4. **View logs (optional):**
   ```bash
   docker-compose logs -f
   ```

5. **Stop services:**
   ```bash
   docker-compose down
   ```

## 📁 File Handling

### Automatic File Collection

Both modes support automatic file discovery and upload:

```python
# The agent will automatically find and upload:
# - CSV files (.csv)
# - Text files (.txt) 
# - JSON files (.json)
# - Python files (.py)
# - Excel files (.xlsx, .xls) - handled by pandas

agent = ReActDataScienceAgent(
    executor="tci",  # or "internal"
    data_dir="/path/to/mixed/data/folder"
)
```

### Supported File Types

- **Text-based**: CSV, JSON, TXT, PY
- **Excel**: XLSX, XLS (processed by pandas)
- **Hidden files**: Automatically excluded
- **Subdirectories**: Recursively scanned

## 🎯 Example Use Cases

```python
# Data exploration
agent = ReActDataScienceAgent(executor="tci", data_dir="./data")
result = agent.run("Explore the data and create a comprehensive EDA report")

# Machine learning
agent.run("""
    Build a predictive model:
    1. Clean and preprocess the data
    2. Split into train/test sets  
    3. Train multiple models and compare performance
    4. Select best model and show feature importance
""")

# Visualization
agent.run("Create publication-ready visualizations showing key insights")

# Statistical analysis
agent.run("Perform statistical tests to validate our hypotheses")
```

## 🔧 Environment Variables

```bash
# Required for TCI mode
export TOGETHER_API_KEY="your-api-key"

# Optional for Docker mode (default: http://localhost:8000)
export CODE_INTERPRETER_URL="http://localhost:8000"
```

## 📝 Notes

- **TCI Mode**: Requires Together AI API key and internet connection
- **Docker Mode**: Requires Docker and docker-compose installed locally
- **File Privacy**: Choose execution mode based on your data sensitivity requirements
- **Performance**: Docker mode offers more consistent performance for compute-intensive tasks
- **Session Persistence**: Both modes maintain state between code executions within a session


