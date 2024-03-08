# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI

class _MyAPI(BundleAPI):

    api_version = 1

    @staticmethod
    def run_provider(session, name, mgr):
        from chimerax.open_command import FetcherInfo, OpenerInfo

        if mgr == session.open_command:
            if 'OME-Zarr' in name:
                class OME_ZarrOpenerInfo(OpenerInfo):
                    check_path = False

                    def open(self, session, path, file_name, **kw):
                        from .open import open_ome_zarr
                        return open_ome_zarr(session, path, **kw)

                    @property
                    def open_args(self):
                        from chimerax.core.commands import IntArg, ListOf, StringArg
                        return {
                            'scales': ListOf(IntArg),
                            'fs': StringArg
                        }

                return OME_ZarrOpenerInfo()


# Create the ``bundle_api`` object that ChimeraX expects.
bundle_api = _MyAPI()