This project includes SFT training of a Llama base model with QLoRA for tool/function calling. The training set is Salesforce/xlam-function-calling-60k and the test set is BFCL_v4_multiple. 


Results on 200 BFCL_v4_multiple test queries are as follows:

| Model                                | Exact match accuracy  |
| ------------------------------------ | --------------------- |
| Salesforce/xLAM-1b-fc-r              | 87.50%                |
| Llama-3.2-3B-Instruct                | 83.00%                |
| Llama-3.2-3B + QLoRA (this)          | 81.50%                |
| Llama-3.2-3B-Instruct + QLoRA (this) | 88.50%                |