import math
import os
import re
import time
import files.porter as porter
from collections import defaultdict


def interactive():
    # 读取文档以及stopword
    script_dir = os.path.dirname(os.path.abspath(__file__))
    documents_path = os.path.join(script_dir, "documents")
    stopwords_path = os.path.join(script_dir, "files", "stopwords.txt")

    documents = read_documents_info(documents_path)
    stopwords = read_stopword_file(stopwords_path)
    p = porter.PorterStemmer()

    index, documents, avg_doclen = create_index(documents, stopwords, p)
    while True:
        query = input("Enter a query (or 'QUIT' to exit): ")
        if query == "QUIT":
            break
        else:
            results = bm25_model(query, documents, index, 1, 0.75, avg_doclen)
            rank = 1
            for result in results:
                print(str(rank) + " " + result[0] + " " + str(result[1]))
                rank += 1
        # Process the query
        # Calculate BM25 scores for documents
        # Sort the documents based on BM25 scores

        # Display the top 15 results
        # for rank, (doc_id, score) in enumerate(sorted_results[:15], start=1):
        #     print(f"Rank: {rank}\tDocument ID: {doc_id}\tSimilarity Score: {score}")


def automatic():
    # 读取文档以及stopword
    load_start_time = time.time()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    documents_path = os.path.join(script_dir, "documents")
    stopwords_path = os.path.join(script_dir, "files", "stopwords.txt")

    documents = read_documents_info(documents_path)
    stopwords = read_stopword_file(stopwords_path)
    p = porter.PorterStemmer()

    index, documents, avg_doclen = create_index(documents, stopwords, p)
    load_end_time = time.time()
    load_time = load_end_time - load_start_time
    print(f"程序加载时间：{load_time}秒")

    print("Loading end")
    start_time = time.time()
    # 打开 "queries.txt" 文件以读取查询
    with open("files/queries.txt", "r") as queries_file:
        queries = queries_file.readlines()

    # 打开 "results.txt" 文件以写入结果
    with open("files/results.txt", "w") as results_file:
        # 遍历每个查询
        for query in queries:
            query_terms = query.strip().split(" ")
            query_id = query_terms[0]
            query_text = clear_txt(" ".join(query_terms[1:]).strip(), stopwords, p)

            # 计算查询与文档的相似度分数，得到排名列表
            ranking = bm25_model(query_text, documents, index, 1, 0.75, avg_doclen)
            rank_number = 1

            # 遍历排名列表，写入结果到 "results.txt" 文件
            for rank in ranking:
                results_file.write(f"{query_id}\t{rank_number}\t{rank[0]}\t{str(rank[1])}\n")
                rank_number += 1

    end_time = time.time()
    runtime = end_time - start_time
    print(f"程序运行时间：{runtime}秒")
    return


def create_index(documents, stopwords, p):
    # 读取索引
    index_file_path = "index.txt"
    if os.path.exists(index_file_path):
        processed_doc = {}
        # 如果索引文件存在，加载索引
        index, avg_doclen = load_index(index_file_path)
        for doc_id, document in documents.items():
            processed_doc[doc_id] = clear_txt(documents[doc_id], stopwords, p)
    else:
        # 如果索引文件不存在，创建索引并保存到文件
        index = {}
        processed_doc = {}
        for doc_id, document in documents.items():
            # 清除标点符号
            document = clear_pun(document)
            words = document.split()
            clean_words = set()
            all_words = []

            for word in words:
                if word not in stopwords:
                    # stemming
                    stem_word = p.stem(word)
                    all_words.append(stem_word)
                    # 创建索引
                    if stem_word not in clean_words:
                        clean_words.add(stem_word)
                        if stem_word in index:
                            index[stem_word]["doc_id"].add(int(doc_id))
                        else:
                            index[stem_word] = {}
                            index[stem_word]["doc_id"] = set()
                            index[stem_word]["doc_id"].add(int(doc_id))
            processed_doc[doc_id] = all_words
        # 存储文档平均长度
        avg_doclen = calculate_avg_doc_len(processed_doc)

        # 计算文档idf
        document_numbers = len(processed_doc)

        for term in index:
            df = len(index[term]["doc_id"])
            idf = math.log((document_numbers - df + 0.5) / (df + 0.5))
            index[term]["idf"] = idf

        save_index(index, "index.txt", avg_doclen)
    return index, processed_doc, avg_doclen


# 清除标点符号以及数字
def clear_pun(document_content):
    clean_document = re.sub(r"[^\w\s]|[\d]", '', document_content)
    return clean_document


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


# 创建索引
def save_index(index, file_path, avg_doclean):
    the_key = "avg_doclean"
    with open(file_path, 'w') as file:
        file.write(f"{the_key}: {avg_doclean}\n")
        for key, value in index.items():
            str_idf = value["idf"]
            str_doc_id = ", ".join(str(item) for item in value["doc_id"])
            file.write(f"{key}: {str_idf}")
            file.write(f", {str_doc_id}\n")


# 读取索引
def load_index(file_path):
    print("Loading BM25 index from file, please wait.")
    index = {}
    with open(file_path, 'r') as file:
        avg_length = file.readline()
        lines = file.readlines()
        avg_length = avg_length.strip().split(': ')[1]
        for line in lines:
            key, value = line.strip().split(': ')
            index[key] = value.split(', ')
    return index, avg_length


# 读取存储文档
def read_documents_info(folder_path):
    document_collection = {}  # 存储文档集合的哈希表

    # 遍历文件夹中的所有文件
    for file_id in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_id)

        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            # 将文字都转换为小写
            document_content = file.read().lower()
            # 将文件内容添加到文档集合中
            if len(document_content) != 0:
                document_collection[file_id] = document_content

    return document_collection


# 读取stopword文件
def read_stopword_file(stopwords_file_path):
    with open(stopwords_file_path, 'r', encoding='utf-8') as file:
        stopwords = file.read().splitlines()
    return stopwords


# BM25 model
def bm25_model(query, documents, index, k, b, avg_doclen):
    scores = {}
    # tf_dict = calculate_term_frequency(documents)
    for term in query:
        if term in index:
            if type(index[term]) is list:
                idf = float(index[term][0])
            else:
                idf = float(index[term]["idf"])
            for doc_id, doc_content in documents.items():
                # tf = tf_dict[doc_id][term]
                tf = doc_content.count(term)
                score = idf * (tf * (k + 1)) / (tf + k * (1 - b + b * (len(doc_content) / float(avg_doclen))))
                if doc_id in scores.keys():
                    scores[doc_id] += score
                else:
                    scores[doc_id] = score
    sorted_documents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_documents[:15]


def calculate_avg_doc_len(documents_clean):
    documents_numbers = len(documents_clean)
    total_word_numbers = 0
    for doc_id, doc_content in documents_clean.items():
        word_numbers = len(doc_content)
        total_word_numbers += word_numbers
    avg_doclen = total_word_numbers / documents_numbers
    return avg_doclen


def calculate_term_frequency(documents):
    tf_dict = {}
    for doc_id, doc_content in documents.items():
        term_fre = defaultdict(int)
        for term in doc_content:
            term_fre[term] += 1
        tf_dict[doc_id] = term_fre
    return tf_dict


# interactive()
automatic()
# print(clear_txt("experimental"))
