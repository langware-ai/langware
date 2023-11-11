from setuptools import setup, find_packages

setup(
    name="Langware",
    version="0.1",
    description="Lightweight, powerful and hassle-free Langchain alternative.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Ivan Stepanov",
    author_email="ivanstepanovftw@gmail.com",
    url="https://github.com/langware-ai/langware",
    packages=find_packages(include=["langware", "langware.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords="langchain openai llama llamacpp chat gpt",
    install_requires=[
        'pydantic >=2, <3',
        'aiohttp >=3, <4',
        'colorama ~=0.4.6',
    ],
    python_requires='>=3.8, <4',
    package_data={
        'Langware': ['LICENSE-APACHE-2.0', 'LICENSE-MIT', 'README.md'],
    },
)
