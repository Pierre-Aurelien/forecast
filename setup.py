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
            "process=gnn.data.process:main",
            "train=gnn.train:main",
            "score=gnn.score:main",
            "random_split=gnn.data.random_split:main",
            "s3_cmd=gnn.s3.s3_cmd:main",
            "analysis_utils=gnn.analysis.analysis_utils:main",
            "analysis_pyrosetta=gnn.analysis.pose_interface_analysis:main",
            "analysis_arpeggio=gnn.analysis.arpeggio.interaction_analysis:main",
        ]
    },
)
