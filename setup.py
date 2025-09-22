from setuptools import setup, find_packages

setup(
    name="tacticore-ai",
    version="0.1.0-alpha",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        # Add more from requirements.txt
    ],
    author="DY, CoreSync",
    description="AI-Powered Mission Planning for Defense & Security",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/coresync-afk/coresync-tacticore-ai",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: License",
        "Operating System :: OS Independent",
    ],
)
