import json
from pathlib import Path
import numpy as np
from keyframe.cli import main
from keyframe.synthetic import write_dicom
from keyframe.config import load_config
import pytest


def test_batch_partial_failure(tmp_path, monkeypatch):
    source = tmp_path/'inputs'
    source.mkdir()
    write_dicom(source/'a.dcm',np.full((5,32,32),1000,np.uint16))
    (source/'b.dcm').write_bytes(b'bad input')
    output = tmp_path/'output'
    monkeypatch.setattr('sys.argv',['batch','--input',str(source),'--output',str(output)])
    assert main('batch') == 1
    result = json.loads((output/'batch_summary.json').read_text())
    assert len(result['samples']) == len(result['errors']) == 1
    assert str(source) not in (output/'batch_summary.json').read_text()
    assert result['evaluation'] is None


@pytest.mark.parametrize('kind',['analyze','batch','benchmark','synthetic'])
def test_help(kind, monkeypatch):
    monkeypatch.setattr('sys.argv',[kind,'--help'])
    with pytest.raises(SystemExit) as exc:
        main(kind)
    assert exc.value.code == 0


def test_config_reject_unknown(tmp_path):
    path = tmp_path/'config.yaml'
    path.write_text('vessel:\n  unknown: 5\n')
    with pytest.raises(ValueError):
        load_config(path)
