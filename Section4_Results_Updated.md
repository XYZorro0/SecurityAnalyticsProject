# 4. Results

## 4.1 Language Distribution in DIFrauD

Language detection revealed that the DIFrauD dataset is predominantly English, with 95,090 samples (99.20%) classified as English. The remaining 764 samples (0.80%) were distributed across 28 non-English languages, with the most common being Afrikaans (78, 0.08%), German (54, 0.06%), French (48, 0.05%), and Dutch (44, 0.05%). An additional 208 samples (0.22%) could not be reliably classified due to insufficient text length or mixed content.

**Table 1: Language Distribution by Domain**

| Domain | Total | English | Non-Eng % | Top Non-English |
|--------|-------|---------|-----------|-----------------|
| SMS | 6,574 | 5,979 | 8.05% | unknown, af, cy |
| Twitter Rumours | 5,789 | 5,733 | 0.97% | de, af, da |
| Political Statements | 12,497 | 12,417 | 0.64% | fr, da, nl |
| Phishing | 15,272 | 15,241 | 0.20% | unknown, it, vi |
| Fake News | 20,456 | 20,456 | 0.00% | — |

The SMS domain exhibited the highest concentration of non-English content at 9.05%, substantially higher than all other domains. This concentration is consistent with the international nature of SMS spam campaigns, which often target multilingual populations. Chi-square analysis revealed a statistically significant association between domain and language (χ² = 223.52, p < 0.001).

## 4.2 Language Distribution by Class

Analysis by class (deceptive versus non-deceptive) revealed an asymmetric distribution: 99.74% of deceptive samples were English compared to 98.86% of non-deceptive samples. The chi-square test confirmed this difference as statistically significant (χ² = 223.52, p = 1.55 × 10⁻⁵⁰). Non-deceptive samples contained 668 non-English instances versus only 96 in the deceptive class, suggesting that legitimate content in the dataset exhibits greater linguistic diversity than fraudulent content.

## 4.3 Classification Performance Comparison

Table 2 presents classification results comparing model performance on the original English-dominated dataset versus the synthetic multilingual dataset. Notably, all models exhibited performance improvements when trained on multilingual data.

**Table 2: Classification Performance Comparison**

| Model | Dataset | Accuracy | Bal. Acc. | F1 (W) | F1 (M) | Change |
|-------|---------|----------|-----------|--------|--------|--------|
| Random Forest | Original | 0.801 | 0.788 | 0.800 | 0.790 | — |
| | Synthetic | 0.806 | 0.794 | 0.806 | 0.796 | +0.7% |
| LinearSVC | Original | 0.795 | 0.797 | 0.797 | 0.789 | — |
| | Synthetic | 0.799 | 0.801 | 0.801 | 0.794 | +0.6% |
| DistilBERT | Original | 0.796 | 0.769 | 0.793 | 0.776 | — |
| | Synthetic | 0.815 | 0.811 | 0.816 | 0.806 | +2.4% |

DistilBERT achieved the highest absolute performance on both datasets and exhibited the largest relative improvement (+2.4% accuracy increase versus +0.6–0.7% for traditional classifiers). The macro F1-score, which equally weights performance across classes, showed DistilBERT improving from 0.776 to 0.806 (3.9% increase), compared to Random Forest's 0.8% increase and LinearSVC's 0.6% increase. These results demonstrate that DistilBERT benefits more substantially from multilingual training data than traditional classifiers.
