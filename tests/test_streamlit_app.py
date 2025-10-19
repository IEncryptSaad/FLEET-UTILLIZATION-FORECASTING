import pytest

streamlit = pytest.importorskip("streamlit")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def test_streamlit_app_bootstraps_without_errors() -> None:
    app = AppTest.from_file("streamlit_app.py")
    app.run(timeout=120)
    assert app.exception is None
