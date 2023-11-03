from setuptools import setup, find_packages

setup(
    name="promptchain",
    version="0.1",
    description="Lightweight, powerful and hassle-free Langchain alternative.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Ivan Stepanov",
    author_email="ivanstepanovftw@gmail.com",
    url="https://github.com/Company-420/promptchain",
    packages=find_packages(include=["promptchain", "promptchain.*", "promptchain_ext", "promptchain_ext.*"]),
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
        'pydantic',
        'aiohttp',
    ],
    extras_require={
        'examples': ['python-dotenv', 'sympy'],
        # 'torch': ['torch'],
        'all': ['examples'],
    },
    python_requires='>=3.8, <4',
    package_data={
        # If you want to include non-python files from your packages (like configuration files, data, etc.)
        'promptchain': ['LICENSE-MIT', 'LICENSE-APACHE', 'README.md'],
    },
    entry_points={
        'console_scripts': [
            'openai_chat=examples.openai_chat:main',
            'openai_extract=examples.openai_extract:main',
        ],
    },
)
