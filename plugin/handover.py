"""The key=value hand-over files the Lightroom plug-in and the Python tools exchange (Lightroom's Lua has no JSON)."""


def read_values(path: str) -> dict[str, str]:
    """The key=value lines of a file as strings (the value may contain '='; blank lines and lines without '=' are skipped)."""
    values = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    return values
