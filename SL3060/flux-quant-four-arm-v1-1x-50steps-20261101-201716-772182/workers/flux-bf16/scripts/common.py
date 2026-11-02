def require(ok, message):
    if not ok: raise ValueError(message)
