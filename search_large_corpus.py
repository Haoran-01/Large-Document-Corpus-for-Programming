import math
import os
import re
import time
import json
import argparse
import files.porter as porter
from collections import defaultdict


# Manual input search
def interactive():
    # Read documents and stopword
    script_dir = os.path.dirname(os.path.abspath(__file__))
    documents_folder = "documents"
    stopwords_path = os.path.join(script_dir, "files", "stopwords.txt")
    stopwords = read_stopword_file(stopwords_path)
    p = porter.PorterStemmer()

    # Create or read index
    index, avg_doclen, tf_dict, len_dict = create_index(documents_folder)

    # Read query and perform search
    while True:
        query = input("Enter a query (or 'QUIT' to exit): ")
        if query == "QUIT":
            break
        else:
            query_text = clear_txt(query, stopwords, p)
            # Search using the bm25 model
            results = bm25_model(query_text, index, 1, 0.75, avg_doclen, tf_dict, len_dict)[:15]
            rank = 1
            for result in results:
                print(str(rank) + " " + result[0] + " " + str(result[1]))
                rank += 1


# Automatic search
def automatic():
    # Read documents and stopword
    load_start_time = time.time()

    documents_folder = "documents"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stopwords_path = os.path.join(script_dir, "files", "stopwords.txt")
    stopwords = read_stopword_file(stopwords_path)
    p = porter.PorterStemmer()

    # Create or load index
    index, avg_doclen, tf_dict, len_dict = create_index(documents_folder)
    load_end_time = time.time()
    load_time = load_end_time - load_start_time
    print(f"Program load time：{load_time}seconds")
    print("Loading end")

    start_time = time.time()
    # Open the "queries.txt" file to read the query
    with open("files/queries.txt", "r") as queries_file:
        queries = queries_file.readlines()

    # Open the "results.txt" file to write the results
    with open("files/results.txt", "w") as results_file:
        # Iterate through each query
        for query in queries:
            query_terms = query.strip().split(" ")
            query_id = query_terms[0]
            query_text = clear_txt(" ".join(query_terms[1:]).strip(), stopwords, p)
            # Calculate the similarity score between the query and the document to get a ranked list
            ranking = bm25_model(query_text, index, 1, 0.75, avg_doclen, tf_dict, len_dict)
            rank_number = 1
            # Iterate through the list of rankings and write the results to the "results.txt" file
            for rank in ranking:
                if rank[1] > 0:
                    results_file.write(f"{query_id}\t{rank_number}\t{rank[0]}\t{str(rank[1])}\n")
                    rank_number += 1
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Program search time：{runtime}seconds")
    return


# Create index on first run, load index later
def create_index(documents_folder):
    # Read the index
    index_file_path = "index.json"
    if os.path.exists(index_file_path):
        print("Loading index")
        # If the index file exists, load the index
        index, avg_doclen, tf_dict, len_dict = load_index(index_file_path)
    else:
        # If index file does not exist, create index and save to file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        stopwords_path = os.path.join(script_dir, "files", "stopwords.txt")

        # Read documents and stopwords
        documents = read_documents_info(documents_folder)
        stopwords = read_stopword_file(stopwords_path)
        p = porter.PorterStemmer()

        # Creating an index
        index = {}
        processed_doc = {}  # Processed article dictionary
        tf_dict = {}  # Word frequency dictionary
        len_dict = {}  # Article length dictionary
        for doc_id, document in documents.items():
            # Clear punctuation
            document = clear_pun(document)
            words = document.split()
            clean_words = set()
            all_words = []
            term_fre = defaultdict(int)
            for word in words:
                if word not in stopwords:
                    # stemming
                    stem_word = p.stem(word)
                    all_words.append(stem_word)  # Storage of processed words
                    term_fre[stem_word] += 1  # Stored word frequency
                    # Create index, index structure is {term : {doc_id : [doc_1, doc_2], idf : idf_value}}
                    if stem_word not in clean_words:
                        clean_words.add(stem_word)
                        if stem_word in index:
                            index[stem_word]["doc_id"].add(doc_id)
                        else:
                            index[stem_word] = {}
                            index[stem_word]["doc_id"] = set()
                            index[stem_word]["doc_id"].add(doc_id)
            # Store word frequency, article length
            tf_dict[doc_id] = term_fre
            len_dict[doc_id] = len(all_words)
            processed_doc[doc_id] = all_words

        # Calculate average document length
        avg_doclen = calculate_avg_doc_len(processed_doc)

        # Calculate document idf
        document_numbers = len(processed_doc)
        for term in index:
            df = len(index[term]["doc_id"])
            idf = math.log((document_numbers - df + 0.5) / (df + 0.5))
            index[term]["idf"] = idf

        # Storing data in a file
        save_index(index, "index.json", avg_doclen, tf_dict, len_dict)
    return index, avg_doclen, tf_dict, len_dict


# Clear punctuation and numbers
def clear_pun(document_content):
    clean_document = re.sub(r"[^\w\s]|[\d]", '', document_content)
    return clean_document


# Process text, remove punctuation, numbers, and stemming
def clear_txt(txt, stopwords, p):
    clean_txt = re.sub(r"[^\w\s]|[\d]", '', txt)
    words = clean_txt.split()
    clean_words = []

    for word in words:
        if word not in stopwords:
            # stemming
            word = p.stem(word)
            clean_words.append(word)
    return clean_words


# Store the index in a json file
def save_index(index, file_path, avg_doclen, tf_dict, len_dict):
    for term, value in index.items():
        value["doc_id"] = list(value["doc_id"])
    # json data format
    data = {
        "avg_doclen": avg_doclen,
        "index": index,
        "tf_dict": tf_dict,
        "len_dict": len_dict
    }
    with open(file_path, 'w', encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


# Read the index file
def load_index(file_path):
    print("Loading BM25 index from file, please wait.")
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    avg_doclen = data['avg_doclen']
    index = data['index']
    tf_dict = data['tf_dict']
    len_dict = data['len_dict']

    return index, avg_doclen, tf_dict, len_dict


# Read document content and perform simple processing
def read_documents_info(documents_folder):
    document_collection = {}  # Hash tables for storing collections of documents
    # Traversing small folders in a large folder
    for folder_name in os.listdir(documents_folder):
        folder_path = os.path.join(documents_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue
        # Traversing files in small folders
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # Read the contents of a file
            with open(file_path, 'r', encoding='utf-8') as file:
                # Convert all text to lower case
                document_content = file.read().lower()
                # Adding document content to a document collection
                if len(document_content) != 0:
                    document_collection[file_name] = document_content
    return document_collection


# Read stopword files
def read_stopword_file(stopwords_file_path):
    with open(stopwords_file_path, 'r', encoding='utf-8') as file:
        stopwords = file.read().splitlines()
    return stopwords


# BM25 model
def bm25_model(query, index, k, b, avg_doclen, tf_dict, len_dict):
    scores = {}  # Dictionary for storing calculated values
    for term in query:
        # Iterate through all the words in the query
        if term in index:
            # Get the idf of the word in the document
            idf = float(index[term]["idf"])
            for doc_id in tf_dict:
                # Iterate through all documents to get document length and word frequency
                doc_len = len_dict[doc_id]
                if term in tf_dict[doc_id]:
                    tf = tf_dict[doc_id][term]
                else:
                    tf = 0
                # Calculate BM25
                score = idf * (tf * (k + 1)) / (tf + k * (1 - b + b * (int(doc_len) / float(avg_doclen))))
                if doc_id in scores.keys():
                    scores[doc_id] += score
                else:
                    scores[doc_id] = score
    sorted_documents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_documents


# Calculate average document length
def calculate_avg_doc_len(documents_clean):
    documents_numbers = len(documents_clean)
    total_word_numbers = 0
    for doc_id, doc_content in documents_clean.items():
        word_numbers = len(doc_content)
        total_word_numbers += word_numbers
    avg_doclen = total_word_numbers / documents_numbers
    return avg_doclen


# main function
def main():
    parser = argparse.ArgumentParser(description='Large Corpus Search Program')
    parser.add_argument('-m', '--mode', choices=['automatic', 'interactive'], required=True,
                        help='Specify the mode to run the program')
    args = parser.parse_args()

    if args.mode == 'automatic':
        automatic()
    elif args.mode == 'interactive':
        interactive()


if __name__ == '__main__':
    main()