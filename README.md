借助Claude code独立开发

  ARAGAS使用方式

  # 检索评估（无需 ragas）
  python main.py --eval --eval-mode retrieval

  # 全链路评估（需先 pip install ragas）
  python main.py --eval --eval-mode all

  # 保存 JSON 报告
  python main.py --eval --eval-output report.json

  # 使用自定义数据集
  python main.py --eval --eval-dataset my_questions.json

  # 调整 Top-K
  python main.py --eval --eval-top-k 10

  ▎ ⚠️ 使用前需确保 .env 已配置（DASHSCOPE_API_KEY、DEEPSEEK_API_KEY）且 chroma_db/ 已建好索引。
