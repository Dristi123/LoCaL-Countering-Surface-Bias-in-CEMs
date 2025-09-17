#!/usr/bin/env python3
import re
import io
import json
import Levenshtein
import javalang  


COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
COMMENT_LINE  = re.compile(r"//.*?$", re.MULTILINE)


TOKEN_PATTERN = re.compile(r"[A-Za-z_]\w*|\d+|\S", re.MULTILINE)

def strip_java_comments(code: str) -> str:
    code = re.sub(COMMENT_BLOCK, "", code)
    code = re.sub(COMMENT_LINE,  "", code)
    return code

def normalize_code_java(code: str) -> str:
   
    code = strip_java_comments(code)
  
    code = re.sub(r"[ \t]+", " ", code)
    code = re.sub(r"\s*\n\s*", "\n", code)
 
    toks = TOKEN_PATTERN.findall(code)
    return " ".join(toks).strip()



def edit_distance_sim(src: str, tgt: str) -> float:
    
    ns = normalize_code_java(src)
    nt = normalize_code_java(tgt)
    max_len = max(len(ns), len(nt))
    if max_len == 0:
        return 1.0
    return 1.0 - (Levenshtein.distance(ns, nt) / max_len)

def ast_similarity_java(src: str, tgt: str) -> float:
  
    try:
        t1 = javalang.parse.parse(src)
        t2 = javalang.parse.parse(tgt)
        n1 = {type(node).__name__ for _, node in t1.filter(javalang.tree.Node)}
        n2 = {type(node).__name__ for _, node in t2.filter(javalang.tree.Node)}
    except Exception:
        return 0.0
    inter = len(n1 & n2)
    union = len(n1 | n2)
    return (inter / union) if union else 0.0

def surface_similarity_java(c1: str, c2: str):
    s_edit = edit_distance_sim(c1, c2)
    s_ast  = ast_similarity_java(c1, c2)
    return {
        "InverseEdit": s_edit,
        "AST": s_ast,
        "SurfaceSim": (s_edit + s_ast) / 2.0
    }

if __name__ == "__main__":
    code1 = """
    public class Main {
        public static void main(String[] args) {
            int a = 5, b = 10;
            System.out.println(a + b);
        }
    }
    """

    code2 = """
    public class Main {
        public static void main(String[] args) {
            int x = 2, y = 3;
            int sum = x + y;
            System.out.println(sum);
        }
    }
    """

    print(json.dumps(surface_similarity_java(code1, code2), indent=2))
