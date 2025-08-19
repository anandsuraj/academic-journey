# 📁 Project Structure Guide

## 🏗️ Repository Organization

```
academic-ai-ml-portfolio/
├── 📄 README.md                              # Main project documentation
├── 📄 PROJECT_STRUCTURE.md                   # This file - structure guide
├── 📄 requirements.txt                       # Python dependencies
├── 📄 .gitignore                            # Git ignore patterns
│
├── 📁 semester-1/                           # Foundation & Core Concepts
│   ├── 📁 game-ai-minimax/                  # Strategic Game AI
│   │   ├── 📄 README.md                     # Project documentation
│   │   ├── 🐍 crossword_game.py            # Main game implementation
│   │   └── 📊 game_analysis.ipynb          # Performance analysis
│   │
│   └── 📁 python-fundamentals/              # Programming Foundations
│       ├── 📄 session1.ipynb               # Basic Python concepts
│       └── 📊 advanced_concepts.ipynb      # Advanced programming
│
├── 📁 semester-2/                           # Advanced AI/ML Applications
│   ├── 📁 deep-neural-networks/             # DNN Architecture & Optimization
│   │   ├── 📄 README.md                     # Comprehensive project guide
│   │   ├── 📊 DNN_Assignment_1_Group_153_final.ipynb  # Main analysis
│   │   ├── 🐍 dnn_tensorFlow.py            # Standalone implementation
│   │   └── 📁 .ipynb_checkpoints/          # Jupyter backup files
│   │
│   ├── 📁 deep-reinforcement-learning/      # DRL Algorithms & Applications
│   │   ├── 📄 README.md                     # Project overview
│   │   ├── 📊 DRL_PS1.ipynb                # Foundational concepts
│   │   ├── 📊 Team115_DP.ipynb             # Dynamic Programming
│   │   │
│   │   ├── 📁 medical-ai-sepsis/           # Healthcare AI Application
│   │   │   ├── 📊 Team_115_ActorCritic.ipynb  # Actor-Critic implementation
│   │   │   └── 🐍 sepsis_environment.py    # Custom medical environment
│   │   │
│   │   └── 📁 drone-surveillance/          # Autonomous Systems
│   │       ├── 📊 Team115_DQNDDQN.ipynb    # DQN/DDQN implementation
│   │       └── 🐍 drone_environment.py     # Custom drone simulation
│   │
│   └── 📁 natural-language-processing/      # NLP & Sentiment Analysis
│       ├── 📄 README.md                     # Detailed methodology
│       ├── 📁 financial-sentiment/          # Financial Text Analysis
│       │   ├── 📄 READ.md                   # Concepts explanation
│       │   ├── 📊 nlp_financial_sentiment_analysis.ipynb  # Main analysis
│       │   ├── 🐍 financial_sentiment_analysis.py  # Standalone script
│       │   ├── 📊 FinancialSentimentAnalysis.csv  # Dataset
│       │   └── 📄 sentiment_analysis_output.md  # Results documentation
│       │
│       └── 📁 venv/                         # Virtual environment (local)
│
└── 📁 environments/                          # Virtual Environment Configs
    └── 📄 environment_setup.md              # Setup instructions
```

## 📋 File Type Legend

| Symbol | Type | Description |
|--------|------|-------------|
| 📄 | Documentation | README files, guides, and documentation |
| 📊 | Jupyter Notebook | Interactive analysis and experimentation |
| 🐍 | Python Script | Standalone executable Python files |
| 📁 | Directory | Folder containing related files |
| 📈 | Data/Results | CSV files, outputs, and analysis results |

## 🎯 Directory Purpose Guide

### **semester-1/** - Foundation Building
**Focus**: Core programming concepts and fundamental AI algorithms

- **game-ai-minimax/**: Demonstrates search algorithms, game theory, and strategic AI
- **python-fundamentals/**: Programming foundations and advanced Python concepts

### **semester-2/** - Advanced Applications
**Focus**: Cutting-edge AI/ML implementations and real-world applications

#### **deep-neural-networks/**
- **Purpose**: Neural network architecture design and optimization
- **Key Skills**: TensorFlow/Keras, regularization, model comparison
- **Deliverables**: Multiple network architectures with performance analysis

#### **deep-reinforcement-learning/**
- **Purpose**: Advanced RL algorithms for real-world problems
- **Key Skills**: Actor-Critic, DQN/DDQN, custom environments
- **Applications**: Healthcare AI, autonomous systems

#### **natural-language-processing/**
- **Purpose**: Text analysis and sentiment classification
- **Key Skills**: Word embeddings, text preprocessing, classification
- **Domain**: Financial text analysis and market sentiment

### **environments/** - Development Setup
**Purpose**: Virtual environment configurations and setup instructions

## 🔧 File Naming Conventions

### **Documentation Files**
- `README.md` - Main project documentation
- `PROJECT_STRUCTURE.md` - Repository organization guide
- `requirements.txt` - Python dependencies

### **Jupyter Notebooks**
- `{Project}_{Team}_{Algorithm}.ipynb` - Team assignment format
- `{domain}_{analysis_type}.ipynb` - Individual project format

### **Python Scripts**
- `{main_functionality}.py` - Descriptive naming
- `{domain}_{specific_task}.py` - Domain-specific scripts

### **Data Files**
- `{DatasetName}.csv` - Dataset files
- `{analysis}_output.md` - Results documentation

## 🎓 Academic Value Mapping

### **Recruiter Perspective**
```
📁 Repository Structure → Professional Organization Skills
📊 Jupyter Notebooks → Interactive Analysis & Research
🐍 Python Scripts → Production-Ready Code
📄 Documentation → Communication & Technical Writing
📈 Results → Problem-Solving & Critical Thinking
```

### **Learning Reinforcement**
```
📁 Organized Structure → Easy Concept Location
📄 Detailed READMEs → Quick Concept Recall
📊 Interactive Notebooks → Hands-on Experimentation
🐍 Clean Code → Implementation Reference
📈 Results Analysis → Performance Understanding
```

## 🚀 Navigation Guide

### **For Recruiters**
1. Start with main `README.md` for overview
2. Browse individual project `README.md` files for technical depth
3. Review Jupyter notebooks for implementation details
4. Check Python scripts for code quality assessment

### **For Concept Recall**
1. Use project-specific `README.md` for quick theory refresh
2. Open relevant Jupyter notebooks for hands-on exploration
3. Reference Python scripts for implementation patterns
4. Review results documentation for performance insights

### **For Development**
1. Check `requirements.txt` for dependency setup
2. Navigate to specific project directories
3. Use Jupyter notebooks for experimentation
4. Refer to documentation for context and methodology

## 🔄 Maintenance Guidelines

### **Adding New Projects**
1. Create appropriately named directory
2. Include comprehensive `README.md`
3. Follow established naming conventions
4. Update main repository `README.md`
5. Add dependencies to `requirements.txt`

### **Documentation Updates**
1. Keep project-specific READMEs current
2. Update main README for new achievements
3. Maintain consistent formatting and style
4. Include performance metrics and results

### **Code Organization**
1. Separate notebooks for exploration vs production scripts
2. Include comprehensive comments and docstrings
3. Follow PEP 8 style guidelines
4. Maintain clean, readable code structure

---

*This structure is designed to showcase academic progression, technical skills, and professional development throughout the M.Tech AI/ML program at BITS Pilani.*