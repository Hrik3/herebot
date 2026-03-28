from importglobal import dataclass,field,Dict,List,Any,Optional,pickle,os
from .directory_handler import STATE_DIR
# ------------------------------------------------------------
# Contextual Sentiment Corpus – emergent learning from user feedback
# ------------------------------------------------------------
@dataclass
class FeedbackEntry:
    """Stores a single feedback instance from user."""
    timestamp: float
    user_input: str
    system_response: str
    factor_dict: Dict[str, float]          # emotional/control state at response time
    rating: str                           # 'positive', 'negative', 'neutral', 'context_positive', 'context_negative', 'context_neutral'
    context_tokens: List[str]             # tokens from user input
    response_tokens: List[str]            # tokens from system response
    response_template_id: Optional[str] = None   # if response came from a template
    metadata: Dict[str, Any] = field(default_factory=dict)
class ContextualSentimentCorpus:
    """
    Maintains a corpus of user feedback with full context.
    Learns sentiment probabilities conditioned on context (tokens, emotional factors, response templates).
    Automatically stops asking for feedback when confidence is high.
    """
    CORPUS_FILE = os.path.join(STATE_DIR, "feedback_corpus.pkl")
    CONFIDENCE_THRESHOLD = 0.85   # stop asking when confidence > threshold
    MIN_SAMPLES = 5               # minimum samples before considering confident

    def __init__(self):
        self.entries: List[FeedbackEntry] = []
        self.context_models = {}   # context_key -> {'pos': Beta(a,b), 'neg': Beta(a,b), 'neu': Beta(a,b)}
        self._load()

    def _load(self):
        if os.path.exists(self.CORPUS_FILE):
            try:
                with open(self.CORPUS_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.entries = data.get('entries', [])
                    self.context_models = data.get('context_models', {})
            except Exception as e:
                print(f"Could not load feedback corpus: {e}")

    def save(self):
        tmp = self.CORPUS_FILE + ".tmp"
        with open(tmp, 'wb') as f:
            pickle.dump({
                'entries': self.entries,
                'context_models': self.context_models
            }, f)
        os.replace(tmp, self.CORPUS_FILE)

    def add_entry(self, entry: FeedbackEntry):
        self.entries.append(entry)
        self._update_model_for_entry(entry)
        self.save()

    def _update_model_for_entry(self, entry: FeedbackEntry):
        """Update Beta posteriors for all relevant contexts."""
        rating = entry.rating.lower()
        if rating not in ['positive', 'negative', 'neutral', 'context_positive', 'context_negative', 'context_neutral']:
            return

        # Use normalized rating: extract base sentiment
        if 'positive' in rating:
            base_sent = 'positive'
        elif 'negative' in rating:
            base_sent = 'negative'
        else:
            base_sent = 'neutral'

        # 1. Token contexts
        for token in entry.context_tokens:
            key = f"token:{token}"
            self._update_context_model(key, base_sent)

        # 2. Emotional factor contexts
        for factor in ['threat', 'trust', 'curiosity', 'energy', 'conscience_active']:
            if factor in entry.factor_dict:
                val = entry.factor_dict[factor]
                if val > 0.7:
                    key = f"factor:{factor}_high"
                elif val < 0.3:
                    key = f"factor:{factor}_low"
                else:
                    key = f"factor:{factor}_mid"
                self._update_context_model(key, base_sent)

        # 3. Response template ID (if available)
        if entry.response_template_id:
            key = f"template:{entry.response_template_id}"
            self._update_context_model(key, base_sent)

    def _update_context_model(self, context_key: str, sentiment: str):
        """Update Beta posterior for a specific context-sentiment pair."""
        if context_key not in self.context_models:
            self.context_models[context_key] = {
                'positive': {'a': 1, 'b': 1},
                'negative': {'a': 1, 'b': 1},
                'neutral': {'a': 1, 'b': 1}
            }
        model = self.context_models[context_key][sentiment]
        if sentiment == 'positive':
            model['a'] += 1
        else:
            model['b'] += 1


    def get_sentiment_probability(self, context_key: str, sentiment: str) -> float:
        """Return posterior mean of P(sentiment | context)."""
        if context_key not in self.context_models:
            return 0.33   # uniform prior
        model = self.context_models[context_key].get(sentiment, {'a':1, 'b':1})
        a = model['a']
        b = model['b']
        return a / (a + b)

    def get_confidence(self, context_key: str) -> float:
        """Return confidence (1 - posterior variance) for the most probable sentiment."""
        if context_key not in self.context_models:
            return 0.0
        max_confidence = 0.0
        total_samples = 0
        for sent, model in self.context_models[context_key].items():
            a = model['a']
            b = model['b']
            total_samples += (a - 1) + (b - 1)   # counts added to prior
            p = a / (a + b)
            variance = (a * b) / ((a + b)**2 * (a + b + 1))
            confidence = 1 - variance
            if confidence > max_confidence:
                max_confidence = confidence
        # Require minimum samples
        if total_samples < self.MIN_SAMPLES:
            return 0.0
        return max_confidence

    def should_ask_feedback(self, tokens: List[str], factor_dict: Dict, template_id: Optional[str] = None) -> bool:
        """
        Determine whether we need more feedback for the current context.
        Returns False if all relevant contexts are confident.
        """
        # Check token contexts
        for token in tokens:
            key = f"token:{token}"
            if self.get_confidence(key) < self.CONFIDENCE_THRESHOLD:
                return True
        # Check factor contexts
        for factor in ['threat', 'trust', 'curiosity', 'energy', 'conscience_active']:
            if factor in factor_dict:
                val = factor_dict[factor]
                if val > 0.7:
                    key = f"factor:{factor}_high"
                elif val < 0.3:
                    key = f"factor:{factor}_low"
                else:
                    key = f"factor:{factor}_mid"
                if self.get_confidence(key) < self.CONFIDENCE_THRESHOLD:
                    return True
        # Check template context
        if template_id:
            key = f"template:{template_id}"
            if self.get_confidence(key) < self.CONFIDENCE_THRESHOLD:
                return True
        return False
