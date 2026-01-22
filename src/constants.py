import enum

class SpecialTokens(enum.Enum):
    START_TOKEN = "<s>"
    END_TOKEN = "</s>"
    PAD_TOKEN = "<pad>"
    UNKNOWN_TOKEN = "<unk>"

START_TOKEN = SpecialTokens.START_TOKEN.value
END_TOKEN = SpecialTokens.END_TOKEN.value
PAD_TOKEN = SpecialTokens.PAD_TOKEN.value
UNKNOWN_TOKEN = SpecialTokens.UNKNOWN_TOKEN.value