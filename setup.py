from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="findgo",
    version="2.0.0",
    description="本地多模态 RAG 图像检索：Chinese-CLIP + Qwen2.5 + 双向量库",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests", "legacy", "scripts")),
    py_modules=["cli"],
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "findgo=cli:main",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
