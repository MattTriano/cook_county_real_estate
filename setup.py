from setuptools import setup, find_packages

setup(
	author="Matt Triano",
    name="myccao",
	description="My Cook County (real estate) Assessing tools.",
    version="0.1.0",
	packages=find_packages(include=["myccao", "myccao.*"]),
	include_package_data=True
)
