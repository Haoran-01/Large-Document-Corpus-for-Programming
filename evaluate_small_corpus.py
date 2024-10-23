def evaluate_results(results_file, qrels_file):
    # 读取结果文件
    with open(results_file, 'r') as f:
        results = f.readlines()

    # 读取相关性判断文件
    with open(qrels_file, 'r') as f:
        qrels = f.readlines()

    # 初始化指标变量
    precision_sum = 0
    recall_sum = 0
    p_at_10_sum = 0
    r_precision_sum = 0
    ap_sum = 0
    num_queries = len(qrels)

    # 处理每个查询的结果
    for query_idx in range(num_queries):
        # 获取查询相关性判断
        relevance_judgments = qrels[query_idx].strip().split()[1:]

        # 获取查询结果
        query_results = results[query_idx].strip().split()

        # 计算指标
        num_relevant = len(relevance_judgments)
        num_retrieved = len(query_results)
        num_relevant_retrieved = 0
        ap = 0

        for idx, doc_id in enumerate(query_results):
            if doc_id in relevance_judgments:
                num_relevant_retrieved += 1
                ap += num_relevant_retrieved / (idx + 1)

        precision = num_relevant_retrieved / num_retrieved if num_retrieved > 0 else 0
        recall = num_relevant_retrieved / num_relevant if num_relevant > 0 else 0
        p_at_10 = num_relevant_retrieved / 10 if num_retrieved >= 10 else 0
        r_precision = num_relevant_retrieved / num_relevant if num_relevant > 0 else 0
        ap /= num_relevant

        # 更新指标总和
        precision_sum += precision
        recall_sum += recall
        p_at_10_sum += p_at_10
        r_precision_sum += r_precision
        ap_sum += ap

    # 计算平均指标
    precision_avg = precision_sum / num_queries
    recall_avg = recall_sum / num_queries
    p_at_10_avg = p_at_10_sum / num_queries
    r_precision_avg = r_precision_sum / num_queries
    ap_avg = ap_sum / num_queries

    # 打印指标的平均值
    print("Precision Avg: {:.4f}".format(precision_avg))
    print("Recall Avg: {:.4f}".format(recall_avg))
    print("P@10 Avg: {:.4f}".format(p_at_10_avg))
    print("R-Precision Avg: {:.4f}".format(r_precision_avg))
    print("MAP Avg: {:.4f}".format(ap_avg))
    print("Bpref Avg: ")  # 如果有bpref计算公式，请在此处进行计算并打印平均值

# 调用函数进行评估
results_file = "files/results.txt"
qrels_file = "files/qrels.txt"
evaluate_results(results_file, qrels_file)
