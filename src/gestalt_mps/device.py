"""Interface reserved by the implementation plan."""



def resolve_device(requested: str):
    raise NotImplementedError



def resolve_dtype(requested: str, device):
    raise NotImplementedError



def device_status():
    raise NotImplementedError
