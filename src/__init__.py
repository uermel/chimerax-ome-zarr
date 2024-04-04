# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):
    api_version = 1

    @staticmethod
    def run_provider(session, name, mgr):
        print(f"run_provider: {name}, {mgr}")

        if mgr == session.open_command:
            if "OME-Zarr" in name:
                from .info import OMEZarrOpenerInfo

                return OMEZarrOpenerInfo()

            elif "ngff" in name:
                from .info import NGFFFetcherInfo

                return NGFFFetcherInfo()


# Create the ``bundle_api`` object that ChimeraX expects.
bundle_api = _MyAPI()
