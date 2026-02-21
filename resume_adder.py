import fitz

def readres(p):
    txt = ""
    with fitz.open(p) as d:
        for pg in d:
            txt += pg.get_text()
    return txt