"""Setup script for python packaging."""
import site
import sys

from setuptools import setup

# enable installing package for user
# https://github.com/pypa/pip/issues/7953#issuecomment-645133255
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

setup(
    name="forecast",
    version="0.1.0",
    description="",
    author="Pierre-Aurelien Gilliot",
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "infer=forecast.run_inference:main",
            "generate=forecast.run_simulation:main",
        ]
    },
)
