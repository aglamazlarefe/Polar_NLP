from pathlib import Path

from setuptools import find_packages, setup

KOK = Path(__file__).parent
gereksinimler = [s.strip() for s in (KOK / "requirements.txt").read_text(encoding="utf-8").splitlines()
                 if s.strip() and not s.startswith("#") and not s.startswith("pytest")]

setup(
    name="polar-nlp",
    version="0.3.0",
    description="TR-KABÜ seyir defteri üslup sınıflandırıcı — BERTurk + Türkçe stilometri hibrit modeli",
    long_description=(KOK / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Efe Ağlamazlar",
    url="https://github.com/aglamazlarefe/Polar_NLP",
    packages=find_packages(include=["polar_nlp", "polar_nlp.*"]),
    install_requires=gereksinimler,
    extras_require={"test": ["pytest>=9.2"]},
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: Turkish",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.10",
)
