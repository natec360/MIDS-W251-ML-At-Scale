# MIDS-W251

# Improving Reddit Post Classification via Deslanging

## Overview
This project investigates the impact of **deslanging**—replacing slang terms with standard English equivalents—on the accuracy of subreddit classification for Reddit self-posts using a variety of machine learning models.

## Authors
Nathan Chiu, Kevin Fu, Allison Schlissel  
📬 [nchiu20@berkeley.edu](mailto:nchiu20@berkeley.edu), [kevinfu@berkeley.edu](mailto:kevinfu@berkeley.edu), [apschlissel@berkeley.edu](mailto:apschlissel@berkeley.edu)  
🔗 [GitHub Repository](https://github.com/apschlissel/w266-final-project)

## Motivation
Reddit is rich in informal language and slang, posing challenges for NLP models. We hypothesized that deslanging Reddit posts could enhance classification accuracy by simplifying and standardizing the input text.

## Research Questions
1. Do transformer-based models outperform a Naive Bayes baseline for subreddit classification?
2. Does deslanging Reddit posts improve model performance?

## Dataset
Reddit self-posts were scraped along with their subreddit labels. Three subreddit group types were analyzed:

- **5 Handpicked Subreddits** (slang-heavy): `wallstreetbets`, `teenagers`, `GenZ`, `copypasta`, `unpopularopinion`
- **5 Similar Subreddits** (gaming-focused): `gaming`, `PS4`, `pokemon`, `xboxone`, `leagueoflegends`
- **5 Random Subreddits**: `Bitcoin`, `memes`, `travel`, `philosophy`, `stocks`

Each group had datasets with 2,500, 5,000, and 25,000 posts.

## Methodology

### Deslanging
- Compiled a slang-to-standard-English dictionary using data from [Slangit.com](https://slangit.com), manually curated for relevance.
- Used regex substitution to replace slang with standardized terms.

### Models Used
- **Naive Bayes (baseline)**: Multinomial model with bag-of-words vectorization.
- **BERT**: Uncased pre-trained transformer model.
- **T5**: Text-to-text transformer reframing classification as generation.
- **RNN**: Two-layer LSTM with dense and softmax layers.

### Evaluation Metric
- Used **F1 Score** to better capture performance across imbalanced classes.

## Results

| Model      | Performance Highlights      | Deslanging Effect          |
|------------|-----------------------------|-----------------------------|
| **BERT**   | Highest F1 (~80%) on Similar Subreddits | Generally **hurt** performance |
| **T5**     | Lower than BERT              | Mixed or negative           |
| **RNN**    | Moderate                     | Minor improvements in some cases |
| **Naive Bayes** | ~60–65% F1 baseline     | Used for comparison         |

## Key Findings
- **Deslanging consistently underperformed** vs. using the original text.
- **BERT outperformed** both T5 and RNN, likely due to contextual bidirectionality.
- **Short posts** and **non-English posts** were misclassified more often.
- **Similar subreddit groups** (e.g. gaming-related) had higher classification errors due to topic overlap.

## Lessons Learned
- Deslanging is error-prone due to:
  - Context-sensitive meanings (e.g., "zzz" = "boring" or "sleeping")
  - Word collisions (e.g., "we" = pronoun or slang for "whatever")
  - Ambiguities in slang like "lol" (game vs. expression)
- Some slang may actually be helpful for classification due to its subreddit-specific usage.

## Future Work
- Develop more **context-aware slang translation models**.
- Fine-tune models on **slang-rich corpora**.
- Investigate **multi-modal signals** (e.g., emojis, formatting) for classification.

## References
See full references in the project report. Notable works:
- Urban Dictionary Embeddings (Wilson et al., 2020)
- SlangSD Sentiment Dictionary (Wu et al., 2018)
- Deep Learning in Social Media NLP (Parvathi, 2021)
- Text Classification with BERT (Winastwan, 2021)

## License
MIT License – see `LICENSE` file for details.
