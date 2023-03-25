import importlib.util
import os
from pathlib import Path

import typer
from loguru import logger

from lcd.utils.setup import abspath  # for monkeypatches

# Define the directory path containing the Python modules to import
apps_dir = abspath() / "apps"
app: typer.Typer = typer.Typer(name="lcd", no_args_is_help=True)


def main():
    """The master entrypoint to lcd.
    :param app: The app to run.
    :type app: str
    """
    command_kwargs = dict(
        context_settings={
            "allow_extra_args": True,
            "ignore_unknown_options": True,
        },
        # no_args_is_help=True
    )
    # Iterate over all files in the directory
    for filename in os.listdir(apps_dir):
        # Check if the file is a Python module
        if filename.endswith(".py"):
            # Construct the module name from the filename
            module_name = filename[:-3]
            # Import the module using importlib
            module_spec = importlib.util.spec_from_file_location(
                module_name, os.path.join(apps_dir, filename)
            )
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            # Add the module to the dictionary
            if hasattr(module, "_app_"):
                module_app = module._app_
                app.add_typer(
                    module_app,
                    name=module_name,
                    **command_kwargs
                    # add docs later by modulestring
                )

            else:
                try:
                    app.command(name=module_name, **command_kwargs)(module.main)
                except AttributeError as e:
                    logger.info(
                        f"{app} doesn't have a main function. Please check that {os.path.join(apps_dir, module_name)}.py defines main()"
                    )

    app()


if __name__ == "__main__":
    main()
