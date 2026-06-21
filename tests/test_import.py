import stereohand


def test_version_present():
    assert isinstance(stereohand.__version__, str)
    assert stereohand.__version__
