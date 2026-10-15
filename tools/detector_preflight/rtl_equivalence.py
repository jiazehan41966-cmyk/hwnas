"""Strict RTL token comparison allowing only bijective HLS internal ID renaming."""
import hashlib
from pathlib import Path
import re


def tokens(text):
    text=re.sub(r'/\*.*?\*/|//[^\n]*','',text,flags=re.S)
    return re.findall(r'[A-Za-z_$][A-Za-z0-9_$]*|\d+|[^\w\s]',text)


def internal_form(name):
    if not re.fullmatch(r'[A-Za-z_$][A-Za-z0-9_$]*',name):return None
    normalized=re.sub(r'(_fu_|_reg_)\d+',r'\1ID',name)
    normalized=re.sub(r'^reg_\d+$','reg_ID',normalized)
    return normalized if normalized!=name else None


def compare_text(left,right):
    a,b=tokens(left),tokens(right)
    if len(a)!=len(b):raise ValueError('RTL token count differs')
    forward={};reverse={}
    for x,y in zip(a,b):
        if x!=y and (internal_form(x) is None or internal_form(x)!=internal_form(y)):
            raise ValueError('RTL differs beyond generated IDs: '+x+' / '+y)
        if internal_form(x) is not None:
            if forward.setdefault(x,y)!=y or reverse.setdefault(y,x)!=x:
                raise ValueError('HLS identifier mapping is not bijective')
    return sum(x!=y for x,y in forward.items())


def compare_rtl(left,right):
    a={p.name:p for p in Path(left).glob('*.v')}
    b={p.name:p for p in Path(right).glob('*.v')}
    if not a or a.keys()!=b.keys():raise ValueError('RTL module sets differ')
    rows=[]
    for name in sorted(a):
        count=compare_text(a[name].read_text(),b[name].read_text())
        rows.append(dict(file=name,renamed_internal_ids=count,
                         left_sha256=hashlib.sha256(a[name].read_bytes()).hexdigest(),
                         right_sha256=hashlib.sha256(b[name].read_bytes()).hexdigest()))
    return dict(status='RTL_TOKEN_IDENTICAL_MODULO_BIJECTIVE_INTERNAL_IDS',files=rows)
