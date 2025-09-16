import ast
import re
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import io, tokenize



TOKEN_PATTERN = re.compile(r'\w+')
TRIPLE_STR = re.compile(r'("""|\'\'\')(?:.|\n)*?\1', re.DOTALL)
LINE_COMMENT = re.compile(r'#.*')

def normalize_code(code: str) -> str:

    try:
        parts = []
        reader = io.StringIO(code).readline
        for tok in tokenize.generate_tokens(reader):
            ttype = tok.type
            tstr  = tok.string
            if ttype == tokenize.COMMENT:
                continue
            if ttype in (tokenize.NL, tokenize.NEWLINE):
                parts.append("\n")
                continue
            if ttype in (tokenize.ENCODING, tokenize.INDENT, tokenize.DEDENT):
                continue
            parts.append(tstr)
        text = "".join(parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()
    except Exception:
       
        text = TRIPLE_STR.sub("", code)
        text = LINE_COMMENT.sub("", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()







def edit_distance_sim(src, tgt):
    max_len = max(len(src), len(tgt))
    if max_len == 0:
        return 1.0
    return 1 - (Levenshtein.distance(normalize_code(src), normalize_code(tgt)) / max_len)
    #return 1-(Levenshtein.distance(src, tgt)/max_len)

def ast_similarity(src, tgt):
    try:
        n1 = {type(node).__name__ for node in ast.walk(ast.parse(src))}
        n2 = {type(node).__name__ for node in ast.walk(ast.parse(tgt))}
    except SyntaxError:
        return 0.0
    inter = len(n1 & n2)
    union = len(n1 | n2)
    return inter / union if union else 0.0

def surface_similarity(c1, c2):
   
    s_edit = edit_distance_sim(c1, c2)
    s_ast = ast_similarity(c1, c2)
    return {
        
        "InverseEdit": s_edit,
        "AST": s_ast,
        "SurfaceSim": (s_edit + s_ast) / 2
    }

if __name__ == "__main__":
   
    code1 = """def is_palindrome(text: str) -> bool:
    for i in range(len(text)):
        if text[i] != text[len(text) - 1 - i]:
            return False
    return True"""


    code2 = """def is_palindrome(text: str) -> bool:
    for i in range(len(text)):
        if text[i] != text[len(text) - 1 - i]:
            return True      
    return False             """

    print(surface_similarity(code1, code2))
