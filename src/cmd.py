# vim: set expandtab shiftwidth=4 softtabstop=4:

# General
import os

# ChimeraX
from chimerax.core import errors
from chimerax.map import Volume
import numpy as np


def envfile_set(session, path: str):
    """Source a file and set the environment variables in the current session."""
    from .util.env import env_from_sourcing
    from .util.settings import OME_ZarrSettings

    if os.path.exists(path):
        settings = OME_ZarrSettings(session, "OME-Zarr")
        env = env_from_sourcing(path)
        for k, v in env.items():
            os.environ[k] = v
        settings.env_file = path
        settings.save("env_file")
    else:
        raise errors.UserError(f"The file {path} does not exist.")


def envfile_get(session):
    """Print the script used to set up the env."""
    from .util.settings import OME_ZarrSettings

    settings = OME_ZarrSettings(session, "OME-Zarr")
    print(f"Env file location: {settings.env_file}")


def envfile_clear(session):
    """Clear the script used to set up the env."""
    from .util.settings import OME_ZarrSettings

    settings = OME_ZarrSettings(session, "OME-Zarr")
    settings.env_file = ""
    settings.save("env_file")


def register_cmds(logger):
    """Register all commands with ChimeraX, and specify expected arguments."""
    from chimerax.core.commands import (
        register,
        CmdDesc,
        StringArg,
    )

    def register_envfile_set():
        desc = CmdDesc(
            required=[("path", StringArg)],
            synopsis="Specify a script to source upon startup.",
        )
        register("envfile set", desc, envfile_set)

    def register_envfile_get():
        desc = CmdDesc(
            synopsis="Print current startup env script.",
        )
        register("envfile get", desc, envfile_get)

    def register_envfile_clear():
        desc = CmdDesc(
            synopsis="Remove startup env script.",
        )
        register("envfile clear", desc, envfile_clear)

    register_envfile_set()
    register_envfile_get()
    register_envfile_clear()
