# JA-Topic-Extractor

# .env

```shell
ENVIRONMENT=development
PORT=3000
LOG_LEVEL=debug
ALLOWED_HOSTS=localhost
MODEL_LANG=english
MODEL_NAME=paraphrase-albert-small-v2
CLASSIFIER_NAME=facebook/bart-large-mnli
CLASSIFIER_TYPE=zero-shot-classification
```

# Model

The lightweight model is `paraphrase-albert-small-v2` so it can be used in development phase. In production
use `all-mpnet-base-v2` for english language and `paraphrase-multilingual-mpnet-base-v2` for multilingual.