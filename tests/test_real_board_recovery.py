import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.sonar_deploy import board_recovery as r


def archive(tmp_path,names=None):
    p=tmp_path/'return.tar.gz'
    if names is None:names=['completion.json','frames.json','layers.json','benchmark.json']
    with tarfile.open(p,'w:gz') as tf:
        for name in names:
            info=tarfile.TarInfo(name);info.size=2
            tf.addfile(info,io.BytesIO(b'{}'))
    return p


def test_extract_complete(tmp_path):
    p=archive(tmp_path);out=tmp_path/'evidence';out.mkdir()
    r.extract_return(p,out,100)
    assert len(list(out.iterdir()))==4


@pytest.mark.parametrize('extra',['../escape','completion.json'])
def test_unsafe_or_duplicate(tmp_path,extra):
    names=['completion.json','frames.json','layers.json','benchmark.json',extra]
    with pytest.raises(ValueError):r.extract_return(archive(tmp_path,names),tmp_path/'evidence',100)
    assert not (tmp_path/'escape').exists()


def test_no_partial_or_overwrite(tmp_path):
    with pytest.raises(ValueError,match='not complete'):r.extract_return(archive(tmp_path,['layers.json']),tmp_path/'evidence',100)
    p=archive(tmp_path);out=tmp_path/'evidence';out.mkdir();(out/'layers.json').write_text('old')
    with pytest.raises(ValueError,match='overwrite'):r.extract_return(p,out,100)
    assert (out/'layers.json').read_text()=='old'


def test_size_bound(tmp_path):
    with pytest.raises(ValueError,match='oversized'):r.extract_return(archive(tmp_path),tmp_path/'evidence',7)


def test_wait_uses_original_return_not_new_board(monkeypatch,tmp_path):
    monkeypatch.setattr(r.c,'remote_json',lambda *args,**kw:dict(return_exists=True,sha256='a'*64))
    value=r.wait_return(dict(remote_project='/project'),dict(build='/build'),tmp_path,5)
    assert value['return_exists']


def test_dead_receiver_fails_closed(monkeypatch,tmp_path):
    monkeypatch.setattr(r.c,'remote_json',lambda *args,**kw:dict(return_exists=False,transport_alive=False))
    with pytest.raises(RuntimeError,match='receiver stopped'):
        r.wait_return(dict(remote_project='/project'),dict(build='/build'),tmp_path,5)


def test_idle_receipt_never_programs_or_frees(monkeypatch,tmp_path):
    out=tmp_path/'evidence';out.mkdir();(out/'completion.json').write_text('{}')
    def command(argv,log):
        script=Path(argv[2]).read_text()
        assert 'download=False' in script and '.download(' not in script
        assert '.write(' not in script and 'freebuffer' not in script
        Path(argv[3]).write_text('RECOVERY_IDLE_RECEIPT '+json.dumps(dict(status='PASS')))
    monkeypatch.setattr(r.c,'_local_command',command)
    r.verify_idle_receipt(tmp_path,dict(board_directory='/board'),out)
    assert json.loads((out/'recovery_idle_receipt.json').read_text())['status']=='PASS'
