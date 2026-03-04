import setuptools
from pathlib import Path

__version__ = "0.0.0"

REPO_NAME = "Enterprise-ML-Pipeline"
AUTHOR_USER_NAME = "asagar-07"
SRC_REPO = "mlPipeline"
AUTHOR_EMAIL = "248957737+asagar-07@users.noreply.github.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="ML pipeline for enterprise applications",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    python_requires=">=3.8",

)