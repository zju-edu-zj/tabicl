"""Read live node-pool queue statistics using a local training rank PID (Linux)."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile

from multiprocessing.managers import BaseProxy, Token
from tabicl.prior._node_pool import _Manager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pid", type=int)
    args = parser.parse_args()
    # Read only the two public rendezvous identifiers; never print environment.
    env = {}
    for field in Path(f"/proc/{args.pid}/environ").read_bytes().split(b"\0"):
        key, _, value = field.partition(b"=")
        if key in (b"TORCHELASTIC_RUN_ID", b"MASTER_PORT"):
            env[key.decode()] = value.decode()
    digest = hashlib.sha256((env.get("TORCHELASTIC_RUN_ID", "") + ":" + env.get("MASTER_PORT", "")).encode()).hexdigest()[:24]
    address = str(Path(tempfile.gettempdir()) / f"tabicl-prior-{digest}.sock")
    manager = _Manager(address=address, authkey=digest.encode())
    manager.connect()
    info = manager._debug_info()
    objects = re.findall(r"([0-9a-f]+):\s+refcount=", info)
    if len(objects) != 1:
        raise RuntimeError(f"expected one live service: {info}")
    proxy = BaseProxy(Token("service", address, objects[0]), "pickle", authkey=digest.encode())
    print(json.dumps(proxy._callmethod("stats"), indent=2))


if __name__ == "__main__":
    main()
