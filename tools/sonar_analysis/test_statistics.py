"""Geometry and malformed-label regression checks; no training."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('sonar_analysis', Path(__file__).with_name('analyze_dataset.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_non_square_original_geometry():
    r = m.bbox_statistics([[0,.5,.5,.1,.2]],200,100,640)[0]
    assert r['width_px'] == pytest.approx(20)
    assert r['height_px'] == pytest.approx(20)
    assert r['normalized_area'] == pytest.approx(.02)
    assert r['aspect_ratio'] == pytest.approx(1)
    assert r['input_width_px'] == pytest.approx(64)
    assert r['input_height_px'] == pytest.approx(128)
    assert r['size_bin'] == 'medium'


@pytest.mark.parametrize('side,expected', [(16,'small'),(32,'medium'),(96,'large')])
def test_size_boundaries(side,expected):
    assert m.bbox_statistics([[0,.5,.5,side/128,side/128]],128,128,128)[0]['size_bin']==expected


@pytest.mark.parametrize('text', ['0 .5 .5 .1', '0 nan .5 .1 .1', '2 .5 .5 .1 .1', '0.5 .5 .5 .1 .1',
    '0 .99 .5 .1 .1', '0 .5 .5 0 .1', '0 .5 .5 .1 .1 0'])
def test_malformed_annotation(tmp_path,text):
    p = tmp_path/'bad.txt'
    p.write_text(text)
    with pytest.raises(ValueError,match='bad.txt:1'):
        m.read_labels(p,2)


def test_empty_vs_missing(tmp_path):
    p = tmp_path/'empty.txt'
    p.write_text('')
    assert m.read_labels(p,2)==[]
    with pytest.raises(FileNotFoundError):
        m.read_labels(tmp_path/'missing.txt',2)


def test_border_and_shape():
    rows = m.bbox_statistics([[0,.1,.1,.2,.2],[0,.5,.5,.4,.1],[0,.5,.5,.1,.4]],100,100,640)
    assert rows[0]['x1']==0
    assert [r['shape'] for r in rows]==['near_square','horizontal','vertical']
