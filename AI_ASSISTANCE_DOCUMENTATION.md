# AI Assistance Documentation

**Project:** Multilingual Data Quality Assessment - DIFrauD Language Analysis
**Course:** COSC 4371 Security Analytics, Fall 2025
**Team:** Joseph Mascardo, Niket Gupta

---

## Overview

This document details how AI tools were used throughout this project, including specific prompts, tasks completed, and the nature of assistance provided.

---

## AI Tools Used

### 1. Claude (Anthropic) - Primary AI Assistant

**Model:** Claude Opus 4
**Usage Period:** Throughout project development

### 2. Grammarly AI

**Usage:** Grammar and spelling corrections in documentation

### 3. Meta NLLB-200 (No Language Left Behind)

**Model:** facebook/nllb-200-distilled-600M
**Usage:** Neural machine translation for synthetic dataset generation

---

## Detailed AI Assistance Log

### Task 1: Synthetic Multilingual Dataset Generation

**Prompt Summary:**
> "Add synthetic multilingual dataset generation to my DIFrauD_LanguageAnalysis__3.ipynb notebook. Use Helsinki-NLP/opus-mt translation models. Translate to Spanish (20%), French (15%), German (10%), Portuguese (8%), Arabic (5%), Chinese (2%). Target 60% non-English, 40% English. Add metadata columns. Save as separate files. Create validation function and DatasetComparator class."

**AI Assistance Provided:**
- Created `MultilingualTranslator` class for handling translations
- Implemented stratified sampling to maintain class balance
- Added metadata columns (`source_language`, `is_translated`, `translation_method`)
- Created `DatasetComparator` class for comparing original vs synthetic datasets
- Generated validation functions for dataset integrity checks

**Code Cells Created:** 8-18 in notebook

---

### Task 2: Translation Optimization

**Prompt Summary:**
> "URGENT: Translation is too slow (1+ hour, only 60% done). Update my notebook with these optimizations: Switch to NLLB-200 model, reduce dataset to 20k samples, batch_size=32, GPU support, progress saving/resume, reduce to 3 languages (Spanish 30%, French 20%, Arabic 10%)"

**AI Assistance Provided:**
- Switched from multiple Helsinki-NLP models to single NLLB-200 model
- Implemented GPU batch processing for faster translation
- Added checkpoint/resume functionality to handle interruptions
- Reduced target languages from 6 to 3 (Spanish, French, Arabic)
- Optimized batch size and added progress tracking

**Result:** Translation time reduced from 1+ hour to ~15 minutes

---

### Task 3: Caching Implementation

**Prompt Summary:**
> "Add checkpoint/caching to the synthetic dataset generation so translation only runs once and can be reused"

**AI Assistance Provided:**
- Added file existence checks at start of translation cell
- Implemented automatic loading from CSV if dataset already exists
- Added cache validation to ensure loaded data matches expected format

**Code Pattern:**
```python
if os.path.exists('difraud_synthetic_multilingual.csv'):
    print("Loading cached synthetic dataset...")
    df_synthetic = pd.read_csv('difraud_synthetic_multilingual.csv')
else:
    # Run translation pipeline
```

---

### Task 4: PowerPoint Presentation Creation

**Prompt Summary:**
> "Using the file 2025Fall4371security_analytics-1.pdf as a guideline, make a .pptx and use the images from the notebook outputs"

**AI Assistance Provided:**
- Created 11-slide presentation using python-pptx library
- Integrated PNG visualizations from notebook outputs
- Structured slides following course guidelines
- Added TODO placeholders for manual image insertion

**File Created:** `DIFrauD_Presentation.pptx`

---

### Task 5: Visualization Improvements for Imbalanced Data

**Prompt Summary:**
> "Currently our notebook creates the same type of 2 visualizations. Some of the visualizations are not able to be analyzed because the data has such a large imbalance and the plot/graph/visualization is not appropriate for the data. Please make appropriate visualizations that are not just bar graphs, but instead are based off the amount of data/balance of the data given."

**AI Assistance Provided:**
- Implemented log-scale charts for minority language visibility
- Added treemaps for proportional language representation
- Created normalized confusion matrices (percentage-based)
- Implemented radar/spider charts for multi-metric model comparison
- Added violin plots for text length distribution
- Created heatmaps with LogNorm color scaling
- Implemented diverging bar charts for performance deltas
- Added slope/dumbbell charts for before/after comparison
- Created parallel coordinates plots for multi-dimensional comparison

**Cells Updated:** 40, 63, 66, 67, 72

---

### Task 6: Visualization Focus Update

**Prompt Summary:**
> "We want the visualizations to only focus on synthetic dataset vs original DIFrauD dataset. I do not want to compare English only vs original DIFrauD full dataset. Update all visualizations accordingly."

**AI Assistance Provided:**
- Changed all visualizations from "Full vs English-only" to "Original DIFrauD vs Synthetic Multilingual"
- Updated all labels, titles, and legends
- Modified cells to use `comparison_results` from DatasetComparator
- Ensured consistent color coding (blue for Original, red/orange for Synthetic)

**Label Changes:**
- "Full (Multilingual)" → "Original DIFrauD"
- "English-only" → Removed
- "Synthetic" → "Synthetic Multilingual"

---

### Task 7: Final Report Writing

**Prompt Summary:**
> "Make a Google Doc about my project. Write as if you're an undergraduate who is new to Cyber Security"

**AI Assistance Provided:**
- Created comprehensive final report in markdown format
- Structured according to academic paper guidelines
- Explained technical concepts in accessible language
- Documented methodology, results, and analysis
- Included proper citations and references
- Added appendix with code availability and AI tools used

**File Created:** `FINAL_REPORT_DIFrauD_Language_Analysis.md`

---

## Code Contributions by AI

### Classes Created:
1. `OptimizedTranslator` - Handles NLLB-200 translation with batching
2. `DatasetComparator` - Compares classifier performance across datasets
3. `DualDatasetAnalyzer` - Helper for running analysis on both datasets

### Functions Created:
1. `create_stratified_sample()` - Stratified sampling maintaining class balance
2. `generate_synthetic_dataset_optimized()` - Main translation pipeline
3. `validate_synthetic_dataset()` - Validation checks for generated data
4. `create_comprehensive_comparison()` - Multi-chart visualization function
5. `create_simple_comparison()` - 2x2 comparison visualization
6. `plot_language_summary_improved()` - Log-scale language distribution
7. `plot_model_performance_radar()` - Radar chart for model metrics
8. `plot_confusion_matrices_normalized()` - Percentage-based confusion matrices

### Visualization Types Implemented:
- Log-scale horizontal bar charts
- Treemaps (using squarify)
- Heatmaps with LogNorm
- Violin plots
- Radar/spider charts
- Dumbbell/slope charts
- Diverging bar charts
- Parallel coordinates
- Bullet charts
- Normalized confusion matrices

---

## Human Contributions

The following tasks were performed by the human team members:

1. **Problem Selection:** Chose the DIFrauD language analysis project
2. **Dataset Acquisition:** Obtained the DIFrauD dataset
3. **Research Direction:** Decided to create synthetic multilingual data
4. **Model Selection:** Specified classifiers (RF, SVM, DistilBERT)
5. **Language Selection:** Chose target languages (Spanish, French, Arabic)
6. **Quality Review:** Validated AI-generated code and results
7. **Execution:** Ran the notebook in Google Colab
8. **Interpretation:** Drew conclusions from results

---

## Prompts Used (Summary)

| Task | Prompt Type | Complexity |
|------|-------------|------------|
| Dataset generation | Feature request | High |
| Translation optimization | Bug fix / optimization | Medium |
| Caching | Feature request | Low |
| Presentation | Content creation | Medium |
| Visualization improvements | Enhancement | High |
| Visualization refocus | Modification | Medium |
| Report writing | Content creation | High |

---

## Ethical Considerations

1. **Transparency:** All AI assistance is documented in this file
2. **Academic Integrity:** AI was used as a tool, not to replace learning
3. **Code Review:** All AI-generated code was reviewed before use
4. **Attribution:** AI tools are credited in the final report

---

## Files Modified/Created by AI

| File | Type | AI Contribution |
|------|------|-----------------|
| `DIFrauD_Language_Analysis_(3)_(1).ipynb` | Notebook | Code cells for translation, visualization |
| `DIFrauD_Presentation.pptx` | Presentation | Full creation |
| `FINAL_REPORT_DIFrauD_Language_Analysis.md` | Report | Full creation |
| `AI_ASSISTANCE_DOCUMENTATION.md` | Documentation | Full creation |

---

## Conclusion

AI assistance significantly accelerated development by handling repetitive coding tasks, implementing complex visualizations, and drafting documentation. However, all critical decisions regarding methodology, model selection, and result interpretation were made by the human team members. The AI served as a productivity tool rather than a replacement for understanding the underlying concepts.

---

*Documentation created for COSC 4371 Security Analytics, University of Houston, Fall 2025*
