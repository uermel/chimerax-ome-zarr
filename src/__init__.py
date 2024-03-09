# vim: set expandtab shiftwidth=4 softtabstop=4:

import os

from chimerax.core.toolshed import BundleAPI
from chimerax.ui.options import InputFileOption


class _MyAPI(BundleAPI):

    api_version = 1

    @staticmethod
    def run_provider(session, name, mgr):
        from chimerax.open_command import FetcherInfo, OpenerInfo

        if mgr == session.open_command:
            if "OME-Zarr" in name:

                class OME_ZarrOpenerInfo(OpenerInfo):
                    check_path = False

                    def open(self, session, path, file_name, **kw):
                        from .open import open_ome_zarr
                        from botocore.auth import NoCredentialsError
                        from .util.settings import OME_ZarrSettings
                        from .util.env import env_from_sourcing, set_env

                        settings = OME_ZarrSettings(session, "OME-Zarr")
                        env = env_from_sourcing(settings.env_file)
                        set_env(env)

                        try:
                            ret = open_ome_zarr(session, path, **kw)
                            return ret
                        except NoCredentialsError:
                            raise NoCredentialsError(
                                """No credentials found for S3. Set an env file with `envfile set` or launch ChimeraX 
                                from terminal."""
                            )

                        return ([], "Error opening file.")

                    @property
                    def open_args(self):
                        from chimerax.core.commands import IntArg, ListOf, StringArg

                        return {"scales": ListOf(IntArg), "fs": StringArg}

                return OME_ZarrOpenerInfo()

    @staticmethod
    def register_command(bi, ci, logger):
        logger.status(ci.name)
        if "envfile" in ci.name:
            from . import cmd

            cmd.register_cmds(logger)


# Create the ``bundle_api`` object that ChimeraX expects.
bundle_api = _MyAPI()
