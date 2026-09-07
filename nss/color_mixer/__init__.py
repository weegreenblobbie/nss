__all__ = ["ColorMixer"]


def __getattr__(name):
    # lazy, so that tests/test_data/make_mixer_tables.py can import lookup.py before tables.py exists
    if name == "ColorMixer":
        from nss.color_mixer.mixer import ColorMixer
        return ColorMixer
    raise AttributeError(name)
