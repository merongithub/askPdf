import fitz  # PyMuPDF
from config import CHUNK_SIZE


def read_pdf(path):
    doc = fitz.open(path)
    text = "\n".join([page.get_text() for page in doc])
    return text


def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


if __name__ == "__main__":
    text = read_pdf("data/sample.pdf")
    chunks = chunk_text(text)
    with open("data/chunks.txt", "w") as f:
        for chunk in chunks:
            f.write(chunk + "\n---\n")
    print(f"Chunked {len(chunks)} sections from PDF.")