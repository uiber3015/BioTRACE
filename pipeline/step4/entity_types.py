"""实体类型定义 - 来自 step2/实体识别.xlsx"""

# 14种实体类型（键为英文，说明为中文）
ENTITY_TYPES = {
    "Gene Symbol": {
        "chinese": "基因名",
        "description": "基因的官方符号或常用名",
        "examples": ["MALAT1", "NEAT2", "2210401K01Rik"]
    },
    "Regulator": {
        "chinese": "调控因子",
        "description": "在调控关系中起上游作用的分子实体",
        "examples": ["NONO", "p53"]
    },
    "Target": {
        "chinese": "靶点",
        "description": "被特定分子调控的下游分子实体",
        "examples": ["TP53"]
    },
    "Functional Mechanism": {
        "chinese": "功能机制",
        "description": "分子发挥功能的途径或机制",
        "examples": ["转录调控", "后转录调控", "表观调控", "翻译调控", "RNA", "SeRNA"]
    },
    "Species": {
        "chinese": "物种",
        "description": "研究中所涉及到的生物体物种",
        "examples": ["人", "小鼠", "大肠杆菌"]
    },
    "Disease": {
        "chinese": "疾病",
        "description": "生物体在生理或心理上出现的异常状态，特指与研究对象（如LncRNA）关联的疾病",
        "examples": ["癌症", "阿尔茨海默病"]
    },
    "Organ": {
        "chinese": "器官",
        "description": "提取所用生物材料来源的器官类型",
        "examples": ["脑", "心脏", "肝脏"]
    },
    "Tissue": {
        "chinese": "组织",
        "description": "提取所用生物材料来源的组织类型",
        "examples": ["皮肤组织", "心肌组织", "神经组织"]
    },
    "Cell": {
        "chinese": "细胞",
        "description": "提取所用生物材料来源的细胞系或细胞类型",
        "examples": ["K562细胞", "HeLa细胞", "T淋巴细胞"]
    },
    "Protein": {
        "chinese": "蛋白质",
        "description": "基因编码的蛋白质产物",
        "examples": ["胰岛素", "血红蛋白"]
    },
    "Compound": {
        "chinese": "化合物",
        "description": "研究中涉及的小分子化合物",
        "examples": ["葡萄糖", "紫杉醇"]
    },
    "Drug": {
        "chinese": "药物",
        "description": "具有治疗作用的化合物或生物制剂",
        "examples": ["阿司匹林", "曲妥珠单抗"]
    },
    "Experimental Method": {
        "chinese": "实验方法",
        "description": "研究中使用的实验技术或方法",
        "examples": ["Western Blot", "CRISPR-Cas9", "RNA-seq"]
    },
    "Phenotype": {
        "chinese": "表型",
        "description": "生物体可观察到的特征或性状",
        "examples": ["肥胖", "抗药性", "细胞凋亡"]
    }
}

# 实体类型列表（英文键）
ENTITY_TYPE_LIST = list(ENTITY_TYPES.keys())


def get_entity_types_description():
    """生成实体类型的详细描述（用于提示词）"""
    lines = []
    for i, (en_name, info) in enumerate(ENTITY_TYPES.items(), 1):
        cn_name = info['chinese']
        desc = info['description']
        examples = ', '.join(info['examples'][:3])
        lines.append(f"  {i}. **{en_name}** ({cn_name}): {desc}。例如：{examples}")
    return '\n'.join(lines)
