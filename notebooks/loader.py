from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader #,PyPDFLoader, PyMuPDFLoader for PDFs in the future
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Loader:
    """ This class handles loading of the documents """

    def __init__(self, path_name: str = "../data", glob: str = "**/*.txt", chunk_size=500, chunk_overlap=100):
        """ """
        self.documents = []
        self.path = Path(path_name)
        self.glob = glob
        self.chunks = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_documents(self):
        """Chunks the text into chunk_size with an overlap specified"""

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.chunks = text_splitter.split_documents(self.documents)
        print(f"Split {len(self.documents)} documents into {len(self.chunks)} chunks.")
                
    def process_files(self): 
        """Processes all files in the path"""

        files = list(self.path.glob(self.glob))
        print(f"Preparing to process {len(files)} text files...")

        for f in files:
            try:
                loader = TextLoader(str(f))
                docs = loader.load()

                for d in docs:
                    d.metadata['manpage'] = Path(f).stem
                    d.metadata['source_file'] = f.name
                    d.metadata['file_type'] = 'txt'
                    d.metadata['platform'] = 'linux'
                    d.metadata['source'] = 'manpage'

                    self.documents.extend(docs)

            except Exception as e:
                print(f"    ERROR {e}")

        print(f"Total documents loaded: {len(self.documents)}.")
        self._split_documents()
