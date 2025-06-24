import os
import nltk
import re
from tqdm import tqdm
nltk.data.path.append('/Users/anniii/nltk_data')

nltk.download('punkt', download_dir='/Users/anniii/nltk_data')


def clean_text(text):
    #  I used these to remove headers, footers, extra spaces
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\f', '', text)
    text = re.sub(r'Page \d+', '', text)
    return text.strip()


def chunk_text(text, chunk_size=250, overlap=50):
    '''
    Args:
        chunk_size: in words (approx)
        overlap: overlapping words between windows
    '''
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > chunk_size:
            chunks.append(' '.join(current_chunk))
            overlap_words = ' '.join(current_chunk).split()[-overlap:]
            current_chunk = [' '.join(overlap_words), sentence]
            current_length = len(' '.join(current_chunk).split())
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


def save_chunks(chunks, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f'==== CHUNK {i} ====\n{chunk}\n')


if __name__ == "__main__":

    infile = 'data/AI Training Document.txt'
    outfile = 'chunks/chunks.txt'

    with open(infile, 'r', encoding='utf-8') as f:
        text = f.read()
    clean = clean_text(text)
    chunks = chunk_text(clean, chunk_size=250, overlap=50)
    save_chunks(chunks, outfile)
    print(f"Saved {len(chunks)} chunks to {outfile}")
