# 假设我们有一个词汇表列表
vocab_list = ['我', '爱', '编程', '学习']

# 创建单词到索引的映射
word2index = {word: index for index, word in enumerate(vocab_list)}
# enumerate() 返回顺序：固定为 (索引, 元素) (0,我)  0 ---> word 我 ---> index
# 创建索引到单词的映射
index2word = {index: word for index, word in enumerate(vocab_list)}
# enumerate() 返回顺序：固定为 (索引, 元素) (0,我)  0 ---> index 我 ---> word

# 打印结果
print("word2index:", word2index)
print("index2word:", index2word)
