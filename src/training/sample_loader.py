"""Sample sentence loader for qualitative evaluation and attention visualization."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class TranslationPair:
    """A source-target translation pair for evaluation."""

    source: str
    target: str


@dataclass
class SampleSentences:
    """Container for sample sentences used in evaluation and attention visualization."""

    evaluation_pairs: list[TranslationPair]
    attention_probes: list[str]

    @classmethod
    def from_yaml(cls, path: Path) -> "SampleSentences":
        """Load sample sentences from a YAML configuration file.

        Args:
            path: Path to the YAML file.

        Returns:
            SampleSentences instance with loaded data.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        pairs = [
            TranslationPair(source=p["source"], target=p["target"])
            for p in data.get("evaluation_pairs", [])
        ]
        probes = data.get("attention_probes", [])

        return cls(evaluation_pairs=pairs, attention_probes=probes)

    @classmethod
    def default(cls) -> "SampleSentences":
        """Return default sample sentences (hardcoded EN-NL pairs).

        These sentences cover various linguistic phenomena for comprehensive
        evaluation of translation quality.
        """
        return cls(
            evaluation_pairs=[
                TranslationPair(
                    source="The meeting was postponed because the report had not been approved.",
                    target="De vergadering werd uitgesteld omdat het rapport nog niet was goedgekeurd.",
                ),
                TranslationPair(
                    source="She only told him after the deadline had passed.",
                    target="Ze vertelde het hem pas nadat de deadline was verstreken.",
                ),
                TranslationPair(
                    source="The keys were eventually found under the table.",
                    target="De sleutels werden uiteindelijk onder de tafel gevonden.",
                ),
                TranslationPair(
                    source="He promised to call back, but he never did.",
                    target="Hij beloofde terug te bellen, maar deed dat nooit.",
                ),
                TranslationPair(
                    source="The contract that the lawyer the client distrusted drafted collapsed.",
                    target="Het contract dat de advocaat die de cliënt wantrouwde opstelde, ging niet door.",
                ),
                TranslationPair(
                    source="Only after the results the committee reviewed were published did they react.",
                    target="Pas nadat de resultaten die de commissie had beoordeeld werden gepubliceerd, reageerden ze.",
                ),
                TranslationPair(
                    source="She picked the children up later than expected.",
                    target="Ze haalde de kinderen later op dan verwacht.",
                ),
                TranslationPair(
                    source="Nobody said that the plan would actually work.",
                    target="Niemand zei dat het plan daadwerkelijk zou werken.",
                ),
                TranslationPair(
                    source="The manager asked whether the system had already been tested.",
                    target="De manager vroeg of het systeem al getest was.",
                ),
                TranslationPair(
                    source="Although he disagreed, he went along with the decision.",
                    target="Hoewel hij het er niet mee eens was, ging hij toch akkoord met de beslissing.",
                ),
            ],
            attention_probes=[
                "Although the old man who the young woman that the doctor trusted criticized smiled politely, the report that he eventually submitted was incomplete.",
                "The contract that the lawyer the client distrusted drafted collapsed.",
                "Only after the results the committee reviewed were published did they react.",
            ],
        )

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "SampleSentences":
        """Load sample sentences from file or return defaults.

        Args:
            path: Optional path to YAML configuration file.
                  If None or file doesn't exist, returns default sentences.

        Returns:
            SampleSentences instance.
        """
        if path is not None and path.exists():
            return cls.from_yaml(path)
        return cls.default()
