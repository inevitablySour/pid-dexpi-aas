"""Without the Cairo library (common on Windows) the app still starts and the check sheet is
still made: only the re-drawn "after" picture is left out, with a note (check_sheet.CAIRO_MISSING)."""
import builtins

from pid_dexpi_aas.render import check_sheet


def test_no_cairo_means_no_after_picture_not_a_crash(monkeypatch):
    real_import = builtins.__import__

    def no_cairo(name, *args, **kwargs):
        if name == "cairosvg":
            raise OSError("no library called 'cairo-2' was found")  # what cairocffi raises without Cairo
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cairo)
    assert check_sheet.render_after_png(None, None) is None
