# Multilingual Data Quality Assessment: Analyzing Language Diversity Impact on DIFrauD Classification Performance

**COSC 4371 Security Analytics - Final Report**
**Fall 2025**

**Team Members:**
- Joseph Mascardo (jamascar@cougarnet.uh.edu)
- Niket Gupta (ngupta21@cougarnet.uh.edu)

**Submission Date:** December 2025

---

## Abstract

This project investigates the impact of language diversity on fraud detection classifier performance using the DIFrauD dataset. We conducted a comprehensive language analysis of the dataset, finding that approximately 99.2% of samples are in English with small pockets of non-English content concentrated in specific domains like SMS (9.14% non-English). To study how multilingual content affects model performance, we created a synthetic multilingual dataset by translating portions of the original data into Spanish, French, and Arabic using Meta's NLLB-200 neural machine translation model. We then compared classifier performance between the Original DIFrauD dataset and our Synthetic Multilingual dataset using Random Forest, Support Vector Machine (LinearSVC), and DistilBERT models. Our results show that DistilBERT achieved the highest performance on the original dataset (F1=0.816) but showed more sensitivity to multilingual content compared to traditional machine learning approaches. This research provides insights for practitioners developing fraud detection systems that must operate in multilingual environments.

---

## 1. Introduction

### 1.1 Problem Statement

Fraud detection is a critical problem in cybersecurity. As online platforms become more global, fraud detection systems need to work with content in many different languages. The DIFrauD dataset is a popular benchmark dataset for training fraud detection models, but we wondered: what languages are actually in this dataset? And how would models perform if they had to deal with more multilingual content?

These questions matter because in the real world, fraudsters don't just operate in English. Phishing emails come in many languages, and scam messages on social media can be in any language. If we train our models only on English data, they might not work well when they encounter other languages.

### 1.2 Research Questions

Our project aimed to answer three main questions:

1. **What is the language composition of the DIFrauD dataset?** We wanted to know how many samples are in English versus other languages, and whether this varies across different fraud domains.

2. **How does multilingual content affect classifier performance?** We wanted to measure whether adding non-English content helps or hurts model accuracy.

3. **Which classifiers are most robust to language diversity?** We compared traditional machine learning models (Random Forest, SVM) with a deep learning approach (DistilBERT).

### 1.3 Approach Overview

To answer these questions, we took the following approach:

1. **Language Detection:** We used the langdetect and spaCy libraries to identify the language of each sample in the DIFrauD dataset.

2. **Synthetic Dataset Creation:** Since the original dataset is mostly English, we created a synthetic multilingual dataset by translating 60% of the samples into Spanish, French, and Arabic using Meta's NLLB-200 model.

3. **Comparative Analysis:** We trained and evaluated three classifiers on both the original and synthetic datasets to measure performance differences.

---

## 2. Background and Related Work

### 2.1 The DIFrauD Dataset

The DIFrauD dataset is a collection of fraud-related text data compiled from multiple public sources. It contains 95,854 samples across seven fraud domains:

| Domain | Samples | Description |
|--------|---------|-------------|
| Fake News | 20,456 | Fabricated news articles |
| Product Reviews | 20,971 | Fake product reviews |
| Phishing | 15,272 | Phishing email content |
| Job Scams | 14,295 | Fraudulent job postings |
| Political Statements | 12,497 | Misleading political content |
| SMS | 6,574 | Spam/scam text messages |
| Twitter Rumours | 5,789 | False claims on social media |

Each sample is labeled as either "deceptive" (fraudulent) or "non-deceptive" (legitimate), making this a binary classification task.

### 2.2 Language Detection in NLP

Language detection is the task of automatically identifying what language a piece of text is written in. Modern language detection libraries like langdetect use character n-gram frequency profiles to classify text. For short texts (like SMS messages), this can be challenging because there isn't much text to analyze.

Previous research by Conneau et al. (2020) showed that language mixing in datasets can hurt model performance, especially for models trained primarily on English data. This motivated our investigation into the DIFrauD dataset's language composition.

### 2.3 Machine Translation for Data Augmentation

Neural machine translation has improved dramatically in recent years. Models like Meta's NLLB-200 (No Language Left Behind) can translate between 200 languages with high quality. We used this model to create our synthetic multilingual dataset because it produces more natural-sounding translations than older statistical methods.

---

## 3. Methodology

### 3.1 Language Detection Pipeline

We implemented a multi-library language detection pipeline to analyze the DIFrauD dataset:

```
1. Primary detection using langdetect library
2. Validation using spaCy's language identification
3. Cross-validation on disagreements
4. Manual review of uncertain cases
```

For each sample, we recorded:
- Detected language code (e.g., "en" for English, "es" for Spanish)
- Confidence score
- Whether the sample is classified as English or non-English

We used a minimum text length threshold of 20 characters because very short texts often produce unreliable language detection results.

### 3.2 Synthetic Multilingual Dataset Generation

Since the original DIFrauD dataset is almost entirely English, we needed a way to study how multilingual content affects model performance. We created a synthetic multilingual dataset using the following approach:

**Translation Configuration:**
- Model: Meta NLLB-200 (facebook/nllb-200-distilled-600M)
- Target languages: Spanish (30%), French (20%), Arabic (10%)
- English retention: 40%
- Sample size: 20,000 samples (stratified sample from original)

**Process:**
1. Take a stratified sample of 20,000 samples from the original dataset
2. Randomly select samples for translation based on the target percentages
3. Translate selected samples using NLLB-200
4. Add metadata columns (source_language, is_translated, translation_method)
5. Validate translations and save the dataset

We chose Spanish, French, and Arabic because they represent different language families and writing systems, which helps us understand how models handle linguistic diversity.

### 3.3 Classification Experiments

We evaluated three classifiers on both datasets:

**1. Random Forest**
- 100 decision trees
- Class weight balancing enabled
- TF-IDF features (max 10,000 features)

**2. Support Vector Machine (LinearSVC)**
- Linear kernel
- Class weight balancing enabled
- TF-IDF features (max 10,000 features)

**3. DistilBERT**
- Pre-trained distilbert-base-uncased
- Fine-tuned for 3 epochs
- Maximum sequence length: 256 tokens

For each experiment, we used an 80/20 train-test split with stratification to maintain class balance.

### 3.4 Evaluation Metrics

We measured model performance using:
- **Accuracy:** Overall percentage of correct predictions
- **Balanced Accuracy:** Average of recall for each class (handles class imbalance)
- **F1 Score (Weighted):** Harmonic mean of precision and recall, weighted by class frequency
- **F1 Score (Macro):** Unweighted average of F1 scores for each class

---

## 4. Results

### 4.1 Language Distribution Analysis

Our language detection analysis revealed that the DIFrauD dataset is predominantly English:

**Overall Statistics:**
- Total samples: 95,854
- English samples: 95,088 (99.20%)
- Non-English samples: 766 (0.80%)

**Language Distribution by Domain:**

| Domain | English % | Non-English % | Top Non-English Languages |
|--------|-----------|---------------|---------------------------|
| Fake News | 100.0% | 0.0% | None |
| Job Scams | 100.0% | 0.0% | None |
| Product Reviews | 99.99% | 0.01% | Italian, Dutch |
| Phishing | 99.78% | 0.22% | Unknown, Italian, Vietnamese |
| Political Statements | 99.36% | 0.64% | French, Danish, Catalan |
| Twitter Rumours | 99.14% | 0.86% | German, Afrikaans, Danish |
| SMS | 90.86% | 9.14% | Unknown, Afrikaans, Somali |

**Key Findings:**
- The SMS domain has the highest non-English content (9.14%), which makes sense because SMS messages are short and may contain slang or abbreviations that confuse language detectors
- Some languages were detected as "unknown" because they were too short or contained special characters
- The Fake News and Job Scams domains are entirely in English

### 4.2 Classification Results on Original Dataset

We first trained and evaluated our classifiers on the original DIFrauD dataset:

| Classifier | Accuracy | Balanced Acc | F1 (Weighted) | F1 (Macro) | Train Time |
|------------|----------|--------------|---------------|------------|------------|
| Random Forest | 80.64% | 79.43% | 0.806 | 0.795 | 78.8s |
| SVM (LinearSVC) | 79.95% | 80.14% | 0.801 | 0.794 | 5.6s |
| DistilBERT | 81.50% | 81.15% | 0.816 | 0.806 | 113.9s |

**Observations:**
- DistilBERT achieved the highest performance across all metrics
- SVM was the fastest to train (5.6 seconds vs 113.9 seconds for DistilBERT)
- All models achieved similar balanced accuracy, suggesting they handle class imbalance reasonably well

### 4.3 Comparison: Original vs Synthetic Multilingual Dataset

After creating our synthetic multilingual dataset, we ran the same experiments to compare performance:

| Classifier | Dataset | F1 (Weighted) | Change |
|------------|---------|---------------|--------|
| Random Forest | Original | 0.806 | - |
| Random Forest | Synthetic | 0.798 | -0.8% |
| SVM (LinearSVC) | Original | 0.801 | - |
| SVM (LinearSVC) | Synthetic | 0.794 | -0.7% |
| DistilBERT | Original | 0.816 | - |
| DistilBERT | Synthetic | 0.789 | -2.7% |

**Key Findings:**

1. **All models showed decreased performance on the multilingual dataset.** This confirms that language diversity introduces challenges for fraud detection.

2. **DistilBERT was most affected by multilingual content.** Despite being the best performer on English data, DistilBERT showed a 2.7% drop in F1 score on the multilingual dataset. This is likely because the base model (distilbert-base-uncased) was pre-trained primarily on English text.

3. **Traditional ML models were more robust.** Random Forest and SVM showed smaller performance decreases (0.7-0.8%), suggesting that TF-IDF features generalize better across languages than transformer embeddings.

4. **Training time was similar across datasets.** The multilingual content didn't significantly increase training time for any model.

---

## 5. Discussion

### 5.1 Implications for Fraud Detection Systems

Our results have several practical implications:

**1. Language matters for model selection.** If your fraud detection system will encounter multilingual content, you might want to consider traditional ML approaches (Random Forest, SVM) rather than transformer models, unless you use a multilingual transformer like mBERT or XLM-RoBERTa.

**2. Data quality assessment is important.** Before training a fraud detection model, it's worth checking the language distribution of your dataset. Hidden non-English content could be affecting your model's performance in unexpected ways.

**3. The 9% non-English content in SMS is significant.** For organizations dealing with SMS fraud detection, this finding suggests that a multilingual approach may be necessary.

### 5.2 Limitations

Our study has several limitations that we should acknowledge:

1. **Synthetic translations may differ from real-world multilingual fraud.** Real fraudsters might use different vocabulary, grammar patterns, or code-switching that our synthetic dataset doesn't capture.

2. **We only tested three target languages.** Other languages (especially non-Latin scripts like Chinese or Arabic) might produce different results.

3. **Small sample size for DistilBERT.** Due to computational constraints, we trained DistilBERT on a subset of the data, which may affect the comparison.

4. **Language detection isn't perfect.** Short texts and unusual vocabulary can confuse language detectors, so our statistics may have some error.

### 5.3 Future Work

Based on our findings, we suggest several directions for future research:

1. **Test multilingual transformers.** Models like mBERT, XLM-RoBERTa, or the multilingual version of DistilBERT might perform better on multilingual fraud data.

2. **Collect real multilingual fraud data.** Creating a benchmark dataset with authentic multilingual fraud content would be valuable for the research community.

3. **Domain-specific analysis.** Since SMS showed the most language diversity, a focused study on multilingual SMS fraud detection could be useful.

4. **Cross-lingual transfer learning.** Investigating whether models trained on English fraud data can detect fraud in other languages would be practically valuable.

---

## 6. Conclusion

In this project, we analyzed the language composition of the DIFrauD dataset and studied how multilingual content affects fraud detection classifier performance. We found that:

1. **The DIFrauD dataset is 99.2% English**, with the SMS domain showing the highest non-English content (9.14%).

2. **Multilingual content decreases classifier performance** across all models we tested.

3. **DistilBERT, while the best performer on English data, was most sensitive to language diversity**, showing a 2.7% drop in F1 score on our synthetic multilingual dataset.

4. **Traditional ML approaches (Random Forest, SVM) showed more robustness** to multilingual content, with only 0.7-0.8% performance drops.

These findings suggest that practitioners developing fraud detection systems for multilingual environments should carefully consider their model selection and potentially use multilingual pre-trained models or ensemble approaches.

---

## 7. References

Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., ... & Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 8440-8451.

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics*, 4171-4186.

NLLB Team. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation. *Meta AI Research*.

Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). Fake news detection on social media: A data mining perspective. *ACM SIGKDD Explorations Newsletter*, 19(1), 22-36.

---

## 8. Appendix

### 8.1 Code and Dataset Availability

All code, datasets, and AI prompts used in this project are available at:
- GitHub Repository: [SecurityAnalyticsProject]
- Jupyter Notebook: `DIFrauD_Language_Analysis_(3)_(1).ipynb`

### 8.2 AI Tools Used

The following AI tools were used in this project:

1. **Claude (Anthropic)** - Used for:
   - Code development assistance for the translation pipeline
   - Debugging and optimization of visualization code
   - Report writing assistance and proofreading

2. **Grammarly AI** - Used for grammatical corrections in documentation

3. **Meta NLLB-200** - Neural machine translation model for creating synthetic multilingual dataset

### 8.3 Dataset Files

| File | Description | Size |
|------|-------------|------|
| difraud_original.csv | Original DIFrauD dataset with language labels | 28.4 MB |
| difraud_synthetic_multilingual.csv | Synthetic multilingual dataset | 18.1 MB |
| classification_results.csv | Model performance metrics | 1 KB |
| language_distribution_by_domain.csv | Language breakdown by domain | 0.6 KB |

### 8.4 Sample Visualizations

The following visualization files were generated:
- `language_distribution_analysis.png` - Language distribution charts
- `classification_results.png` - Model comparison visualizations
- `comprehensive_dataset_comparison.png` - Original vs Synthetic comparison
- `model_side_by_side.png` - Side-by-side model performance
- `presentation_confusion_matrices.png` - Confusion matrix comparison

---

*Report prepared for COSC 4371 Security Analytics, University of Houston, Fall 2025*
