import argparse, json, random
from pathlib import Path
import numpy as np, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq, set_seed
from peft import LoraConfig, get_peft_model
from common import prompt_for, target_for

def read_jsonl(path): return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/qwen_lora.json"); a=ap.parse_args()
    c=json.loads(Path(a.config).read_text()); set_seed(c["seed"]); random.seed(c["seed"]); np.random.seed(c["seed"])
    tok=AutoTokenizer.from_pretrained(c["model_name"]); tok.pad_token=tok.eos_token
    rows=read_jsonl("data/processed/train.jsonl"); val=read_jsonl("data/processed/validation.jsonl")
    trunc={"train":[],"validation":[]}
    def encode(r, split):
        p=prompt_for(r); t=target_for(r)
        pids=tok(p,add_special_tokens=False)["input_ids"]; tids=tok(" "+t+tok.eos_token,add_special_tokens=False)["input_ids"]
        max_prompt=max(1,c["max_length"]-len(tids))
        was=len(pids)>max_prompt
        pids=pids[:max_prompt]; tids=tids[:c["max_length"]-len(pids)]
        trunc[split].append({"patient_id":r["patient_id"],"truncated":was,"input_tokens_before":len(tok(p,add_special_tokens=False)["input_ids"]),"input_tokens_used":len(pids)})
        ids=pids+tids; return {"input_ids":ids,"attention_mask":[1]*len(ids),"labels":[-100]*len(pids)+tids}
    train_ds=Dataset.from_list([encode(r,"train") for r in rows]); eval_ds=Dataset.from_list([encode(r,"validation") for r in val])
    model=AutoModelForCausalLM.from_pretrained(c["model_name"],torch_dtype="auto")
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=c["lora_r"],lora_alpha=c["lora_alpha"],lora_dropout=c["lora_dropout"],target_modules=["q_proj","k_proj","v_proj","o_proj"],task_type="CAUSAL_LM"))
    out=Path(c["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    (out/"truncation.json").write_text(json.dumps(trunc,indent=2),encoding="utf-8")
    args=TrainingArguments(output_dir=str(out),num_train_epochs=c["epochs"],learning_rate=c["learning_rate"],per_device_train_batch_size=c["per_device_train_batch_size"],per_device_eval_batch_size=1,gradient_accumulation_steps=c["gradient_accumulation_steps"],warmup_ratio=c["warmup_ratio"],weight_decay=c["weight_decay"],logging_steps=5,eval_strategy="epoch",save_strategy="epoch",save_total_limit=2,load_best_model_at_end=True,metric_for_best_model="eval_loss",greater_is_better=False,gradient_checkpointing=True,fp16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0]<8,bf16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0]>=8,report_to="none",seed=c["seed"])
    trainer=Trainer(model=model,args=args,train_dataset=train_ds,eval_dataset=eval_ds,data_collator=DataCollatorForSeq2Seq(tok,padding=True,label_pad_token_id=-100))
    trainer.train(); trainer.save_model(str(out/"best_adapter")); tok.save_pretrained(str(out/"best_adapter")); (out/"train_metrics.json").write_text(json.dumps(trainer.state.log_history,indent=2),encoding="utf-8")
if __name__=="__main__": main()
