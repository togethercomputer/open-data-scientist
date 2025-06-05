
## 📊 Evaluation

The ReAct Data Science Agent includes two comprehensive evaluation frameworks to test its capabilities on real-world data science tasks:

### 🎯 DABstep Evaluation

**DABstep** is a domain-specific evaluation focusing on data analysis with complex, multi-step reasoning requirements.

#### Features
- **Automatic data setup**: Data is pre-downloaded during Docker build
- **Domain expertise**: Specialized prompts for financial transaction analysis  
- **Complex reasoning**: Multi-step questions requiring deep understanding of domain concepts
- **Factoid Q&A**: Tests precise numerical and categorical answer generation

#### Setup and Usage

The DABstep evaluation requires no manual data setup - everything is handled automatically:

```bash
# Run the full DABstep evaluation
cd eval
python dabstep.py

# Test with just the first few examples
python dabstep.py --test-first-only

# Skip hard difficulty questions
python dabstep.py --skip-hard

# Submit results (creates submission file)
python dabstep.py --submit --which-split dev
```

#### How it works
1. **Docker pre-loads data**: The `interpreter/Dockerfile` automatically downloads DABstep dataset files during build
2. **Specialized prompts**: The agent receives detailed instructions about financial domain concepts from `manual.md`
3. **Precise file paths**: Uses absolute paths like `/app/downloaded_data/data/context/payments.csv`
4. **Domain validation**: Emphasizes reading domain manuals before analysis to ensure correct interpretations

### 🏆 Kaggle Competition Evaluation

**Kaggle** evaluation tests the agent's ability to solve real machine learning competitions end-to-end.

#### Features
- **End-to-end ML**: Complete pipeline from data exploration to submission generation
- **Multiple competition types**: Supports different ML tasks (NLP, classification, etc.)
- **Manual data setup**: Realistic workflow requiring data acquisition
- **Submission generation**: Creates properly formatted competition submissions

#### Setup and Usage

Kaggle evaluation requires manual setup of data and API credentials:

1. **Setup Kaggle API credentials**:
   ```bash
   # Place your kaggle.json file in ~/.kaggle/
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

2. **Download competition data**:
   ```bash
   # For Jigsaw Toxic Comment Classification
   cd eval/kaggle_data/jigsaw
   chmod +x download_jigsaw_data.sh
   ./download_jigsaw_data.sh
   
   # For other competitions, follow similar pattern
   ```

3. **Run evaluation**:
   ```bash
   cd eval
   
   # Run Jigsaw competition
   python kaggle.py --competition jigsaw
   
   # Run Spooky Author competition  
   python kaggle.py --competition spooky
   ```

#### Supported Competitions
- **jigsaw**: Toxic Comment Classification Challenge
- **spooky**: Spooky Author Identification  
- More competitions can be added by extending the `COMPETITIONS` mapping

#### How it works
1. **Manual data acquisition**: Users download competition data using provided scripts
2. **General ML prompts**: Flexible instructions that work across different competition types
3. **Dynamic paths**: Uses competition-specific data directories like `/app/jigsaw`
4. **Submission focus**: Emphasizes creating properly formatted `submission.csv` files

### 🔄 Key Differences

| Aspect | DABstep | Kaggle |
|--------|---------|---------|
| **Data Setup** | ✅ Automatic (Docker build) | 🔧 Manual (download scripts + API) |
| **Domain Focus** | 💰 Financial data analysis | 🤖 General ML competitions |
| **Task Type** | ❓ Factoid Q&A with precise answers | 📈 End-to-end ML model building |
| **Prompt Style** | 📚 Highly detailed domain instructions | 🎯 Flexible, competition-agnostic |
| **File Paths** | 📍 Fixed absolute paths | 🗂️ Dynamic competition-specific |
| **Validation** | ✅ Ground truth comparison | 📊 Competition leaderboard |
| **Prerequisites** | Docker only | Docker + Kaggle API + manual downloads |

### 🛠️ Custom Prompt Behavior

Both evaluations demonstrate the agent's ability to handle specialized prompting:

- **DABstep**: Domain-specific instructions emphasizing manual reading and financial terminology
- **Kaggle**: Competition-focused prompts that adapt to different ML problem types

This showcases how the ReAct framework can be customized for different domains while maintaining the same underlying reasoning and acting capabilities.