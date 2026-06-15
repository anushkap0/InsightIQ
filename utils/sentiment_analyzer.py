import torch
from transformers import pipeline
from typing import Optional, Callable


class SentimentAnalyzer:
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    ):
        self.model_name = model_name
        self._pipe      = None

        if torch.cuda.is_available():
            self._device = 0
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = -1   

    
    def _load(self):
        if self._pipe is None or self._pipe.model.name_or_path != self.model_name:
            self._pipe = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=self._device,
                truncation=True,
                max_length=512,
            )

    def analyze_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict:
        """
        Run sentiment analysis in batches.

        Returns:
            {"sentiments": [...], "confidences": [...]}
        """
        self._load()

        sentiments:  list[str]   = []
        confidences: list[float] = []
        total = len(texts)

        for start in range(0, total, batch_size):
            chunk   = texts[start : start + batch_size]
            results = self._pipe(chunk, truncation=True, max_length=512)

            for r in results:
                sentiments.append(r["label"])
                confidences.append(round(r["score"], 4))

            if progress_callback:
                progress_callback(min(start + batch_size, total), total)

        return {"sentiments": sentiments, "confidences": confidences}

    
    def analyze(self, texts: list[str], **kwargs) -> dict:
        return self.analyze_batch(texts, **kwargs)