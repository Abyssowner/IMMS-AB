import torch
from model import ClassifierNet

def predict_single_sequence(heavy_chain, light_chain, antigen, checkpoint_path):
    """
    对单个抗体-抗原序列对预测亲和力
    
    参数:
        heavy_chain (str): 重链氨基酸序列
        light_chain (str): 轻链氨基酸序列
        antigen (str): 抗原氨基酸序列
        checkpoint_path (str): 模型检查点路径
    
    返回:
        float: 预测的亲和力值
    """
    # 1. 加载预训练模型
    model = ClassifierNet.load_from_checkpoint(checkpoint_path)
    model.eval()  # 设置为评估模式
    
    # 2. 准备输入数据
    # 假设你有一个分词器来处理序列
    from transformers import EsmTokenizer
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    
    # 对序列进行编码
    heavy_tokens = tokenizer(heavy_chain, return_tensors="pt")["input_ids"].to(model.device)
    light_tokens = tokenizer(light_chain, return_tensors="pt")["input_ids"].to(model.device)
    antigen_tokens = tokenizer(antigen, return_tensors="pt")["input_ids"].to(model.device)
    
    # 3. 进行预测
    with torch.no_grad():  # 不计算梯度
        prediction = model(heavy_tokens, light_tokens, antigen_tokens)
    
    # 4. 返回预测结果
    return prediction.item()

# 使用示例
if __name__ == "__main__":
    # 示例序列
    heavy_chain = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDLGWSDSYYYYYGMDVWGQGTTVTVSS"
    light_chain = "DIQMTQSPSSLSASVGDRVTITCRASQGIRNDLGWYQQKPGKAPKRLIYAASSLQSGVPSRFSGSGSGTEFTLTISSLQPEDFATYYCLQDYNYPWTFGQGTKVEIK"
    antigen = "MKTPITEAIAAADTLQSLDSHAKGIVQVIDVHEGYLASVGDTFLNTPKTNIQKTEIRLLREMNYADLPCLOLHLGLDGKKITLQNGDTETPDYPITVSSNATCTDAFCNNDISCVTIMVPKSLVSKPYSWLKKENKGVNVFSNTGDMANFIQDNVLIPLVVLSDSTLCTDENYKENLYFQGSHHHHHH"
    
    # 模型检查点路径
    checkpoint_path = "lightning_logs/version_1/checkpoints/epoch=29-step=10000.ckpt"
    
    # 预测亲和力
    affinity = predict_single_sequence(heavy_chain, light_chain, antigen, checkpoint_path)
    print(f"预测的亲和力值: {affinity:.4f}")
