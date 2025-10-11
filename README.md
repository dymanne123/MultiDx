Code implementation for MultiDx.

First, obtaining predictions and evidence from different knowledge sources. We implemented the web search module based on deep research and modify the web search settings to block any websites that may potentially cause data leakage. 
Run:

```bash
python MedReason_RAG.py --model deepseek-R1 --num_shot 10

python MedReason_RAG_trace.py --model deepseek-R1 --num_shot 10

python MedReason_SOAP.py --model deepseek-R1 --num_shot 10

python MedReason_web_search.py --model deepseek-R1 --num_shot 10
```

Then we obtain the final answer:
```bash
python MultiDx.py --model deepseek-R1 --num_shot 10
```

To evaluate the accuracy:
```bash
python Evaluation_list.py --folder_path "../results/MedReason_deepseek-R1_10shot_all" --hit 1,5,10
``

To evaluate the reasoning recall:
```bash
python LLM_eval.py --model deepseek-R1 --num_shot 10
```
