# setup.py
from setuptools import setup, find_packages
from setuptools.command.install import install


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


install_requires = [
    "datasets==2.17.1",
    "evaluate==0.4.1",
    "nltk==3.8.1",
    "levenshtein==0.25.0",
    "sacremoses==0.1.1",
    "sentence-splitter==1.4",
    "tqdm==4.65.0",
    "torch==2.2.0",
    "transformers==4.40.0",
    "faiss-cpu==1.8.0",
    "bitsandbytes==0.43.1",
]
dependency_links = []


class PostInstall(install):
    @staticmethod
    def post_install():
        """Post installation `nltk` downloads."""
        import nltk

        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('universal_tagset')


    def run(self):
        install.run(self)
        self.execute(
            PostInstall.post_install, [], msg="Running post installation tasks"
        )


setup(
    name="cute",
    version="0.1.0",
    author="anonymous",
    description="A benchmark for evaluating LLMs' understanding of their tokens",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="anonymous",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=install_requires,
    dependency_links=dependency_links,
    cmdclass={"install": PostInstall},
)