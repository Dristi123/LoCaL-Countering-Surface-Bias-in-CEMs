import re
from collections import Counter
from nltk.util import ngrams
from nltk.translate.bleu_score import SmoothingFunction
from crystalbleu import corpus_bleu
from pygments.token import Comment
from pygments.lexers import JavaLexer, CppLexer, PythonLexer


def get_lexer(language: str):
    """Returns the appropriate Pygments lexer based on the language name."""
    language = language.lower()
    if language == 'java':
        return JavaLexer()
    elif language in ('cpp', 'c++'):
        return CppLexer()
    elif language == 'python':
        return PythonLexer()
    else:
        raise ValueError(f"Unsupported language: {language}")


def tokenize(code: str, lexer) -> list:
    
    return [tok[1] for tok in lexer.get_tokens(code)
            if not (re.fullmatch(r'\s+', tok[1]) or tok[0] in Comment)]


def crystal_BLEU(references: list, candidates: list, language: str, top_k: int = 50) -> float:
  
    if len(references) != len(candidates):
        raise ValueError("Number of references and candidates must be the same.")

    lexer = get_lexer(language)
    sm_func = SmoothingFunction().method1

    tokenized_refs = []
    tokenized_cands = []
    all_ngrams = []

    for ref, cand in zip(references, candidates):
        try:
            ref_tokens = tokenize(ref, lexer)
            cand_tokens = tokenize(cand, lexer)
        except Exception as e:
            print(f"Tokenization error: {e}")
            continue

        tokenized_refs.append([ref_tokens])  # reference should be wrapped in a list
        tokenized_cands.append(cand_tokens)

        tokenzied_corpus=ref_tokens+cand_tokens
        
        for n in range(1, 5):
            all_ngrams.extend(ngrams(tokenzied_corpus, n))
    
    freq = Counter(all_ngrams)
    most_common_dict = dict(freq.most_common(top_k))

    score = corpus_bleu(
        tokenized_refs, tokenized_cands,
        smoothing_function=sm_func,
        ignoring=most_common_dict
    )
    return score
