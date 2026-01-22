from sacrebleu import CHRF, corpus_bleu

class Metrics():
    def __init__(self):
        pass
    
    def bleu(self, references: list[str], hypotheses: list[str]) -> float:
        # 1. Use references and hypotheses to compute BLEU score
        bleu = corpus_bleu(hypotheses, references)
        return bleu.score
    
    def chrf(self, references: list[str], hypotheses: list[str]) -> float:
        # 1. Use references and hypotheses to compute chrF score
        chrf = CHRF()
        score = chrf.corpus_score(hypotheses, [references])
        return score.score