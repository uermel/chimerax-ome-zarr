# vim: set expandtab shiftwidth=4 softtabstop=4:

import os
import subprocess
import sys
from typing import Dict


# On Macs the environment of applications can be very different from that in shells, so this convenience function can
# be used to source a file and return the environment that results from sourcing it.
# From: https://stackoverflow.com/a/47080959
def env_from_sourcing(file_to_source_path, include_unexported_variables=False):
    a = {}
    if os.path.isfile(file_to_source_path):
        command = f"source {file_to_source_path} && env"
        for line in subprocess.getoutput(command).split("\n"):
            kv = line.split("=")
            k = kv[0]
            v = kv[1] if len(kv) > 1 else ""
            a[k] = v

    return a


def set_env(values: Dict[str, str]):
    for k, v in values.items():
        os.environ[k] = v


def env_if_mac():
    if sys.platform == "darwin":
        set_env(env_from_sourcing(f"{os.path.expanduser('~')}/.zprofile"))
