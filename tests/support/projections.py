from collections.abc import Iterator

def route_fields(value: object, path: str = "$") -> Iterator[str]:
    names = {"expert_flagged", "expert_audit", "cleared", "route.v1", "route-v1"}
    if isinstance(value, dict):
        for key, item in value.items():
            if "route" in key.lower() or "routing" in key.lower():
                yield f"{path}.{key}"
            yield from route_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from route_fields(item, f"{path}[{index}]")
    elif isinstance(value, str) and value in names:
        yield f"{path}: {value}"
