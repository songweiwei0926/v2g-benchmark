# V2G-Benchmark-OneShot v1.0
## Enhancer Variant → Target Gene Prediction Unified Benchmark
### 完整冻结工程规范 / Codex 执行文档

---

# 0. 项目最终目标

本项目建立一个统一、可复现、一次性执行完成的：

> **Regulatory Variant-to-Gene Prediction Benchmark**

核心问题定义为：

给定一个非编码调控变异 \(v\)、生物学 context \(c\)，以及该位点附近候选基因集合：

\[
G(v,c)=\{g_1,g_2,\ldots,g_n\}
\]

要求模型为每个候选基因计算：

\[
S(v,g,c)
\]

并将真实受调控基因排在前面。

最终评价：

\[
variant \rightarrow target\ gene
\]

而不是仅仅：

\[
variant\rightarrow functional/nonfunctional
\]

也不是仅仅：

\[
enhancer\rightarrow gene
\]

---

# 1. 项目必须满足的硬性原则

以下规则为 **NON-NEGOTIABLE**。

## 1.1 一次执行

项目最终必须能够通过唯一主命令：

```bash
bash run_once.sh
```

自动完成：

```text
Preflight
↓
资源锁定
↓
数据下载
↓
数据校验
↓
数据标准化
↓
benchmark 构建
↓
candidate gene 构建
↓
所有 published predictions 下载
↓
所有本地模型推理
↓
AlphaGenome API 推理
↓
模型统一
↓
指标计算
↓
bootstrap
↓
分层分析
↓
failure analysis
↓
ensemble
↓
Main Figures
↓
Supplementary Figures
↓
Tables
↓
最终 QC
↓
release package
```

正式运行过程中不得要求人工：

- 改 Python；
- 改 R；
- 改 threshold；
- 改文件名；
- 手动删除失败样本；
- 手工匹配 cell type；
- 手工重新拼表；
- 手工选择模型；
- 手工调整画图数据。

---

# 2. “一次跑完”的准确含义

允许在运行前仅完成一次机器配置：

```text
HPC_ROOT
SLURM_PARTITION_CPU
SLURM_PARTITION_GPU
ALPHAGENOME_API_KEY
SYNAPSE_AUTH_TOKEN
GITHUB_TOKEN（可选）
```

这些只能放：

```text
.env
```

或者：

```text
config/site.yaml
```

不能写进代码。

运行后：

```bash
bash run_once.sh
```

如果 preflight 不通过：

**正式计算不得开始。**

因此不存在：

> GTEx 已经跑完了才发现 Borzoi 装不上。

---

# 3. Benchmark 不允许混成一个数据集

整个 benchmark 分为 5 个独立 Track。

---

# Track A：Direct Variant-to-Gene

最核心任务。

真实单位：

```text
variant
gene
context
```

Gold standard 主要来源：

```text
GTEx v11 fine-mapped eQTL
eQTL Catalogue
GWAS variant-gene benchmark
```

回答：

> 这个 variant 到底调哪个 gene？

主指标：

```text
MRR
Top-1
Recall@3
Recall@5
NDCG
```

---

# Track B：Experimental Enhancer-to-Gene

来源：

```text
CRISPRi
CRISPRa
Perturb-seq
FlowFISH
```

这里没有具体 causal nucleotide。

单位：

```text
enhancer
gene
cell_type
```

回答：

> enhancer 到底调哪个 gene？

用于检验：

```text
ABC
ENCODE-rE2G
scE2G
GraphReg
EpiMap
...
```

不能谎称它是 direct V2G gold standard。

---

# Track C：Disease Variant/Locus-to-Gene

来源：

```text
fine-mapped GWAS
GWAS silver standards
Open Targets L2G
```

回答：

> disease-associated locus 最可能影响哪个 gene？

单独报告。

---

# Track D：Variant Effect

来源：

```text
TraitGym
```

回答：

\[
variant\rightarrow regulatory/nonregulatory
\]

这不是 V2G。

作用是分析：

> variant-effect prediction 强  
> 是否意味着 V2G prediction 也强？

TraitGym 已经公开 dataset、predictions、features 和 chromosome-aware metrics，可以作为 frozen auxiliary benchmark。

---

# Track E：Integrated V2G

最后训练：

```text
Variant Effect
+
E2G
+
Distance
+
Sequence model
```

形成：

\[
S_{integrated}(v,g)
\]

这是本文最后的方法学部分。

---

# 4. 模型纳入原则

本项目禁止无限加入所谓“所有 genomic model”。

“全部模型”严格定义为：

> 截止项目冻结日期，具有公开代码、公开 prediction、公开权重或稳定官方接口，并且能够直接或经过明确转换输出 enhancer/variant → gene score 的模型。

以下情况不作为 Main V2G model：

1. 只能预测 variant pathogenicity；
2. 完全不能输出 gene-specific score；
3. 必须重新训练才能完成任务；
4. 没有公开实现；
5. 无公开权重或 predictions；
6. 只能预测 promoter variant；
7. 模型输出与 target gene 无法建立无歧义转换。

所有被排除的方法必须自动写进：

```text
results/tables/model_exclusions.tsv
```

字段：

```text
model
reason
reference
code_available
weights_available
gene_specific
included
```

---

# 5. 正式模型面板

模型分 6 类。

---

## Family 0：Negative control / simple baselines

必须包含：

```text
Random
Nearest-TSS
Inverse-distance
Exponential-distance-50kb
Exponential-distance-100kb
Exponential-distance-250kb
Nearest-expressed-gene
```

其中：

\[
Score_{nearest}=-distance
\]

\[
Score_{inverse}=\frac{1}{d+1}
\]

\[
Score_{\exp}=\exp(-d/\lambda)
\]

\[
\lambda =
50kb,\ 100kb,\ 250kb
\]

这些不需要训练。

---

# Family 1：Classical E2G

至少包含：

```text
ABC
ENCODE-rE2G
scE2G-ATAC
scE2G-Multiome
EpiMap
GraphReg
```

ABC 的基本思想是：

\[
ABC(E,G)
=
\frac{Activity_E\times Contact_{E,G}}
{\sum_{E'} Activity_{E'}\times Contact_{E',G}}
\]

官方 ABC pipeline 当前公开并维护。

ENCODE-rE2G 是在 ABC 基础上建立的 logistic-regression E2G pipeline；为复现 2026 ENCODE-rE2G paper，应冻结论文所用 **release 1.0.0**，而不是未来最新版。

scE2G 官方实现能够基于 single-cell ATAC 或 multiome，结合 ABC、E2G features 和单细胞 correlation 输出 E–G score。

EpiMap 提供了约 3.3 million tissue-specific enhancer–gene links，是一个重要历史 comparator。

GraphReg 使用 sequence、1D epigenomic data 和 3D chromatin contact 建模 gene regulation。

---

# Family 2：Single-cell E2G

必须包含：

```text
pgBoost
SCENT
Signac
ArchR
Cicero
```

优先直接使用作者公开的 frozen predictions。

pgBoost 论文已经把：

```text
pgBoost
SCENT
Signac
ArchR
Cicero
```

的 linking scores / percentiles 一起公开到 Zenodo，所以本项目**禁止重新跑 raw multiome 来复现这五个模型作为第一选择**。

论文中使用的软件版本记录为：

```text
SCENT 0.1.0
Signac 1.10.0
ArchR 1.0.2
Cicero 1.3.9
XGBoost 1.7.5.1
```

必须记录进 provenance。

---

# Family 3：Sequence-to-expression / sequence model

必须包含：

```text
AlphaGenome
Borzoi
Enformer
```

这三个必须真正执行模型推理，而不只是抄论文指标。

---

## AlphaGenome

正式使用：

```text
recommended RNA_SEQ GeneMaskLFC scorer
```

即：

\[
Score_{signed}
=
\log(Expression_{ALT}+0.001)
-
\log(Expression_{REF}+0.001)
\]

gene ranking 使用：

\[
Score_{rank}=|Score_{signed}|
\]

AlphaGenome 官方当前已经提供 gene-centric RNA-seq variant scorer。

保存：

```text
alphagenome_signed
alphagenome_abs
alphagenome_quantile
```

---

## Borzoi

下载官方 4 replicate：

```text
replicate_0
replicate_1
replicate_2
replicate_3
```

官方明确公开四个 human model replicates。

每个 variant：

REF inference

ALT inference

对每个 candidate gene：

按 gene exons 聚合 RNA signal。

replicate ensemble：

\[
S_{Borzoi}
=
\frac{1}{4}
\sum_{r=1}^{4}
S_r
\]

保存：

```text
borzoi_signed
borzoi_abs
borzoi_sd_across_replicates
```

---

## Enformer

使用官方 DeepMind implementation / official model weights。

Enformer 没有 Borzoi/AlphaGenome 那么直接的 exon-level RNA output。

因此冻结转换方法：

对 candidate gene 的 canonical TSS：

```text
TSS ± 1 kb
```

选 context-matched human CAGE tracks。

计算：

\[
S=
\log_2(CAGE_{ALT}+1)
-
\log_2(CAGE_{REF}+1)
\]

多个 matched tracks：

取平均。

保存：

```text
enformer_signed
enformer_abs
```

绝对值用于 target ranking。

---

# Family 4：Disease-specific gene prioritization

仅用于 Track C：

```text
OpenTargets-L2G
PoPS
ABC-Max
```

Open Targets L2G 与普通 V2G 必须严格区分。

L2G 使用：

```text
distance
molecular QTL
chromatin interaction
variant pathogenicity
```

等证据进行 locus gene prioritization。

Open Targets 原始 gold standard 为：

```text
445 positive loci/genes
```

且采用 chromosome-aware outer CV。

当前 Open Targets L2G 已有方法更新，因此必须分别标记：

```text
OT-L2G-2021
OT-L2G-current
```

不能混为同一模型。

---

# Family 5：自动发现的 published comparators

这是本工程保证“尽量不漏模型”的关键。

ENCODE-rE2G paper benchmark 已经公开 comparison pipeline，其 GWAS benchmark 仓库说明：

> paper 中多个预测方法的 E2G prediction bundle 可从 Synapse 获取，全部数据约 76 GB。

因此建立：

```text
discover_encode_models.py
```

它必须读取作者：

```text
methodsTable
predictionsTable
```

自动发现所有模型 / configuration。

任何满足：

```text
chrom
start
end
gene
score
```

的 prediction configuration：

全部加入 Supplementary benchmark。

不得人工只挑喜欢的模型。

---

# 6. Main Figure 模型与 Supplementary 模型

因为 ENCODE bundle 可能包含大量模型 configuration：

Main Figure 只展示代表性方法：

```text
Nearest-TSS
ABC
ENCODE-rE2G
scE2G
EpiMap
GraphReg
pgBoost
SCENT
Signac
ArchR
Cicero
Enformer
Borzoi
AlphaGenome
Integrated-V2G
```

但：

```text
Supplementary Table
```

必须报告自动发现的：

**全部 configuration。**

不能因为 Main Figure 太拥挤而丢掉数据。

---

# 7. Gold-standard 数据

---

# Dataset A1：ENCODE / Engreitz CRISPR benchmark

下载：

```text
EngreitzLab/CRISPR_comparison
```

重点文件：

```text
resources/crispr_data/
EPCrisprBenchmark_ensemble_data_GRCh38.tsv.gz
```

官方 workflow 明确用于将 E–G prediction 与 CRISPR experimental results 进行比较。

禁止重新从原始论文整理。

---

# Dataset A2：Held-out CRISPR

下载：

```text
EngreitzLab/ENCODE_Test_Dataset_Analysis
```

其 workflow：

```text
Differential Expression
SCEPTRE
Power Analysis
High-confidence negatives
Data Integration
```

已经全部完成设计。

直接使用作者处理结果。

---

# Dataset B1：GTEx v11 SuSiE

官方文件：

```text
GTEx_Analysis_v11_eQTL_SuSiE.tar
```

GTEx Portal 当前列出其大小约：

```text
174.1 MB
```

且 GTEx v11 使用 GENCODE 47。

同时下载：

```text
gencode.v47.genes.gtf
```

---

# Dataset B2：eQTL Catalogue

正式 stable benchmark：

```text
stable release
```

同时增加：

```text
r8 beta extension
```

但必须分开报告。

eQTL Catalogue r8 在 2026 年进入 pre-release，并正在继续重新处理旧数据，所以 beta 数据不能冒充 stable final release。

因此：

```text
eQTLCatalogue_STABLE
eQTLCatalogue_R8_BETA
```

是两个 dataset ID。

官方提供：

```text
metadata
association files
fine-mapping results
```

下载接口。

---

# Dataset C1：pgBoost published benchmark

Zenodo：

```text
11211925
```

直接获取：

```text
pgBoost scores
existing method scores
eQTL evaluation
ABC evaluation
CRISPR evaluation
GWAS SNP-gene evaluation
```

pgBoost 论文公开 benchmark 包含：

```text
4,434 fine-mapped eSNP-eGene links
53,701 ABC SNP-gene links
892 CRISPR links
155 GWAS-derived noncoding SNP-gene links
```

这些非常适合做 pipeline reproduction sanity check。

---

# Dataset C2：GWAS E2G silver standard

直接使用：

```text
EngreitzLab/GWAS_E2G_benchmarking
```

其中已经包含：

```text
fine-mapped GWAS
credible-set gene linking
silver-standard causal gene data
```

这个 pipeline 已经定义了：

```text
variant enrichment
credible-set → causal gene
```

两种 GWAS benchmark。

禁止自己重新从 GWAS Catalog 整理。

---

# Dataset C3：Open Targets

直接 clone：

```text
opentargets-archive/genetics-gold-standards
```

使用：

```text
gold_standards/processed/
```

其 repository 已经整理 >400 高可信 GWAS loci，并提供 high / medium / low confidence。

Main：

```text
high + medium
```

Sensitivity：

```text
high only
```

---

# Dataset D：TraitGym

直接使用：

```text
songlab/TraitGym
```

下载：

```text
datasets
features
predictions
metrics
```

仅作为 variant-effect benchmark，不混进主 V2G label。

---

# 8. 不允许重新人工整理 Gold Standard

整个项目严禁：

```text
人工读 Supplementary Table
人工复制 Excel
人工决定哪个 gene 是 target
人工根据论文文字判断 positive
```

允许的人工行为只有：

> 运行脚本。

任何数据转换必须由：

```text
adapter
```

完成。

---

# 9. 项目目录结构

必须严格使用：

```text
v2g-benchmark/
│
├── ENGINEERING_SPEC.md
├── README.md
├── pyproject.toml
├── environment.yml
├── run_once.sh
├── Snakefile
│
├── config/
│   ├── project.yaml
│   ├── site.yaml
│   ├── datasets.yaml
│   ├── models.yaml
│   ├── benchmarks.yaml
│   ├── context_mapping.yaml
│   ├── resources.yaml
│   └── plotting.yaml
│
├── workflow/
│   ├── rules/
│   │   ├── preflight.smk
│   │   ├── download.smk
│   │   ├── reference.smk
│   │   ├── harmonize.smk
│   │   ├── benchmarks.smk
│   │   ├── models_published.smk
│   │   ├── models_sequence.smk
│   │   ├── evaluate.smk
│   │   ├── ensemble.smk
│   │   └── figures.smk
│   │
│   └── envs/
│       ├── core.yaml
│       ├── borzoi.yaml
│       ├── enformer.yaml
│       ├── alphagenome.yaml
│       └── r.yaml
│
├── src/
│   └── v2gbench/
│       ├── io/
│       ├── schemas/
│       ├── harmonize/
│       ├── benchmark/
│       ├── models/
│       ├── metrics/
│       ├── statistics/
│       ├── plotting/
│       └── utils/
│
├── scripts/
│   ├── preflight/
│   ├── download/
│   ├── harmonize/
│   ├── benchmark/
│   ├── models/
│   ├── evaluate/
│   └── figures/
│
├── tests/
│
├── data/
│   ├── raw/
│   ├── locked/
│   ├── interim/
│   ├── processed/
│   └── reference/
│
├── benchmarks/
│   ├── crispr/
│   ├── eqtl/
│   ├── gwas/
│   ├── l2g/
│   └── common/
│
├── predictions/
│   ├── published/
│   ├── sequence/
│   ├── baseline/
│   └── integrated/
│
├── logs/
│
└── results/
    ├── metrics/
    ├── bootstrap/
    ├── stratified/
    ├── failures/
    ├── figures/
    ├── tables/
    └── release/
```

---

# 10. 文件格式原则

大表禁止 CSV。

内部统一：

```text
Parquet + ZSTD
```

只有最终：

```text
TSV
Excel
```

供人阅读。

原因：

- 文件更小；
- 类型稳定；
- Polars/DuckDB 快；
- 适合 HPC。

---

# 11. Variant canonical schema

所有 variant 最终必须为：

```text
variant_id
chrom
pos
ref
alt
genome_build
rsid
```

canonical ID：

```text
GRCh38:chr1:123456:A:G
```

主键：

```text
variant_id
```

禁止 rsID 做主键。

---

# 12. Variant normalization

必须经过：

```text
bcftools norm
```

并使用：

```text
GRCh38 FASTA
```

执行：

```bash
bcftools norm \
  -f GRCh38.fa \
  -m -any \
  input.vcf.gz
```

所有：

```text
REF mismatch
```

必须写：

```text
qc_variant_status
```

允许：

```text
PASS
REF_MISMATCH
MULTIALLELIC_SPLIT
LIFTOVER_FAILED
INVALID_ALLELE
```

REF_MISMATCH 不允许进入 benchmark。

---

# 13. Gene canonical schema

统一：

```text
gene_id
gene_symbol
chrom
start
end
strand
tss
gene_type
```

主键：

```text
Ensembl Gene ID
```

必须去版本：

```text
ENSG000001234.12
```

转换：

```text
ENSG000001234
```

annotation：

```text
GENCODE v47
```

---

# 14. Gene master table

生成：

```text
data/reference/gene_master.parquet
```

必须包含：

```text
gene_id
gene_symbol
chrom
tss
strand
start
end
gene_type
canonical_transcript
exon_intervals
```

Sequence model 必须使用同一份 gene definitions。

---

# 15. Context canonical schema

所有：

```text
K562
Liver
liver tissue
Whole Blood
PBMC
CD4 T cell
```

不能直接字符串比较。

统一：

```text
context_id
context_name
context_type
ontology_id
parent_context
supergroup
```

ontology：

```text
CL
UBERON
EFO
```

---

# 16. 自动 context mapping

必须实现：

```text
map_contexts.py
```

顺序：

### Level 1

exact ontology ID。

### Level 2

normalized exact string。

### Level 3

synonym。

### Level 4

ontology parent / child distance <= 1。

### Level 5

ontology distance <= 2。

### Level 6

published scE2G fine/coarse mapping。

scE2G analysis 已公开 eQTL Catalogue：

```text
tissue_fine
tissue_coarse
```

mapping，应直接复用而不是人工重做。

每次映射输出：

```text
source_context
target_context
mapping_method
ontology_distance
mapping_confidence
```

Primary benchmark：

```text
mapping_confidence >= 0.8
```

Supplementary：

全部。

---

# 17. Gold evidence schema

统一：

```text
benchmark_id
evidence_id
variant_id
element_id
gene_id
context_id
trait_id
evidence_type
label
effect_size
effect_direction
pip
pvalue
source_dataset
source_publication
confidence
training_overlap
```

---

# 18. Evidence type 枚举

只允许：

```text
CRISPRi
CRISPRa
PerturbSeq
FlowFISH
eQTL
GWAS
curated_L2G
```

不得出现 20 种拼写。

---

# 19. 不要真正删除重复 evidence

建立两个表。

## evidence_long.parquet

每个独立来源保留。

例如：

```text
variant v1
gene MYC
GTEx
```

以及：

```text
variant v1
gene MYC
eQTL Catalogue
```

分别保存。

---

## canonical_pairs.parquet

聚合 biological pair：

```text
variant_id
gene_id
context_id
```

增加：

```text
n_evidence_sources
evidence_sources
max_confidence
```

这样可以分析：

> 多个独立 evidence 支持的 V2G 是否更容易预测？

---

# 20. GTEx 和 eQTL Catalogue 去重

必须识别：

```text
GTEx in eQTL Catalogue
```

Main eQTL-Catalogue benchmark：

```text
exclude study == GTEx
```

GTEx 独立作为：

```text
GTEX_V11
```

禁止重复计数。

---

# 21. eQTL positive 定义

Main：

```text
PIP >= 0.90
```

Sensitivity：

```text
PIP >= 0.50
PIP >= 0.70
PIP >= 0.95
```

不得后来看到结果不好再改 cutoff。

所有四套阈值一次性计算。

---

# 22. eQTL negative 定义

不能把“没有 PIP ≥0.9”的所有 pair 当 negative。

严格 negative：

```text
variant tested for gene
AND
same context
AND
PIP <= 0.01
AND
not member of meaningful credible set
```

其它：

```text
label = unknown
```

AUPRC/AUROC：

只用：

```text
positive + confident negative
```

Ranking：

candidate universe 全部使用。

---

# 23. Candidate gene universe

这是项目最重要规则之一。

Primary：

\[
variant\pm1Mb
\]

内：

```text
context-expressed / context-tested genes
```

---

# 24. Candidate gene sensitivity

自动同时生成：

```text
250 kb
500 kb
1 Mb
```

生成：

```text
candidate_250k.parquet
candidate_500k.parquet
candidate_1m.parquet
```

Main：

```text
1 Mb
```

因为 eQTL Catalogue cis mapping 本身以邻近 cis window 为基本框架。

---

# 25. Expression filtering

GTEx：

gene 必须在相应 tissue 的 tested expression dataset 中。

eQTL Catalogue：

使用 phenotype metadata 中 tested gene。

如果 dataset 无法获得 expression/tested list：

fallback：

```text
all GENCODE genes ±1Mb
```

并设置：

```text
candidate_basis = GENCODE_FALLBACK
```

不能 silently fallback。

---

# 26. Candidate pair schema

```text
candidate_set_id
variant_id
gene_id
context_id
distance_to_tss
distance_rank
is_nearest
is_gold
gold_confidence
candidate_basis
```

---

# 27. Gold target distance rank

每个 gold gene 必须计算：

```text
gold_distance_rank
```

例如：

```text
1
2
3
4+
```

这是 Main Figure 必须使用的字段。

---

# 28. CRISPR benchmark

CRISPR E2G 不得重新生成 candidate genes。

只能评价：

```text
experimentally tested pairs
```

negative 优先使用：

```text
authors' negative
powered negative
```

不能自己定义：

```text
P > 0.05 = negative
```

held-out pipeline 已经进行了 power analysis，所以优先使用作者 high-confidence negatives。

---

# 29. Model prediction 统一 schema

无论原模型是什么：

最终必须输出：

```text
model_id
model_family
benchmark_id
variant_id
element_id
gene_id
context_id
raw_score
ranking_score
signed_score
coverage
applicability
source_mode
```

其中：

```text
higher ranking_score = stronger predicted link
```

永远统一。

---

# 30. Source mode

只能是：

```text
published_prediction
local_inference
remote_inference
derived_baseline
derived_ensemble
```

---

# 31. Published predictions 优先原则

优先级：

```text
1. 作者公开 frozen prediction
2. 官方 pretrained model
3. 官方代码 + 官方 checkpoint
4. 禁止自行重写论文模型
```

也就是说：

### pgBoost / SCENT / Signac / ArchR / Cicero

优先：

```text
Zenodo published linking scores
```

而不是重新跑 single-cell raw data。

### ABC / rE2G / GraphReg / EpiMap

优先：

```text
ENCODE published comparison predictions
```

而不是自己从 BAM 开始。

这样最大限度避免：

> 我们自己 implementation 和论文不一样。

---

# 32. E2G → V2G 转换

E2G model：

\[
Score(E,G)
\]

variant：

\[
v
\]

如果：

\[
v\in E
\]

则：

\[
Score(v,g)=Score(E,g)
\]

如果一个 variant overlap 多个 elements：

Main：

\[
Score(v,g)
=
\max_E Score(E,g)
\]

Supplementary：

额外计算：

```text
sum
mean
```

但 Main 固定 max。

---

# 33. variant 没落在 model enhancer 怎么办

不能删除。

定义：

```text
coverage = 0
ranking_score = 0
```

实际应用 benchmark：

使用 0。

Common-coverage analysis：

只使用：

```text
coverage == 1
```

两套结果同时计算。

---

# 34. Tie breaking

模型产生相同 score 时：

依次：

```text
1. score descending
2. distance ascending
3. gene_id alphabetical
```

这样结果完全 deterministic。

---

# 35. AlphaGenome 推理策略

这是绝对不能写错的地方。

官方明确说明 AlphaGenome API 适合：

```text
smaller / medium-scale analyses
1000s of predictions
```

而不是 >1 million predictions。

因此严禁：

```text
每个 variant-gene pair 调一次 API
```

必须：

```text
一个 variant-context
↓
一次 score_variant
↓
返回多个 gene-centric scores
↓
本地筛 candidate genes
```

AlphaGenome API 本身支持 gene-centric score 输出。

---

# 36. Sequence Core Benchmark

为保证 AlphaGenome、Borzoi、Enformer 在完全相同样本上公平比较：

建立：

```text
SEQ_CORE
```

最多：

```text
5,000 unique variant-context pairs
```

如果总数量 <5000：

全用。

如果 >5000：

使用 deterministic stratified selection。

---

# 37. SEQ_CORE 抽样方法

禁止真正随机抽。

使用：

```text
SHA256(variant_id + context_id)
```

得到稳定 hash。

在以下 strata 内按 hash 排序选取：

```text
benchmark_source
distance_bin
PIP_bin
nearest/non-nearest
chromosome
context_supergroup
```

保证不同证据均有覆盖。

任何电脑重新跑：

选择完全一致。

---

# 38. 为什么 Sequence Model 不在百万 pair 上跑

Sequence model ranking 的单位仍然是：

```text
5,000 variants
×
所有 candidate genes
```

例如平均 20 gene：

已经：

```text
100,000 V2G pairs
```

但 AlphaGenome 远程 inference 只做：

```text
5,000 variant-context calls
```

而不是 100,000 calls。

这是 frozen design，不是跑不动后的临时缩减。

---

# 39. Borzoi 推理

每个 variant：

构造 REF / ALT sequence。

使用 4 replicates。

GPU sharding：

```text
one chromosome / shard
```

输出：

```text
predictions/sequence/borzoi/
chr1.parquet
...
chr22.parquet
```

每个 shard 完成后产生：

```text
.done
```

---

# 40. Borzoi OOM 自动处理

禁止人工改 batch。

程序：

```text
initial batch = 8
```

若 OOM：

```text
8 → 4 → 2 → 1
```

自动执行。

如果 batch=1 仍失败：

该 shard 标记 fatal。

---

# 41. AlphaGenome rate limit

实现：

```text
HTTP retry
exponential backoff
checkpoint
resume
```

建议：

```text
retry:
1 min
2 min
4 min
8 min
...
```

但设置最大 backoff。

每成功一个 batch：

立即写 parquet。

不得把所有结果存在 RAM 等最后再保存。

---

# 42. AlphaGenome cache key

```text
model_version
variant_id
context_ontology
scorer
```

生成 hash。

如果 cache 已存在：

绝不重复调用。

---

# 43. Enformer

使用 GPU。

输入 REF / ALT。

对 context-compatible CAGE tracks 聚合。

如果没有匹配 track：

```text
applicability = NOT_APPLICABLE_CONTEXT
```

不是：

```text
score = 0
```

---

# 44. Applicability 和 Coverage 必须区分

### applicability=FALSE

模型从定义上不能预测该 context。

### coverage=FALSE

模型适用，但没有输出 link。

这是完全不同的概念。

---

# 45. Model-context applicability matrix

运行前生成：

```text
results/tables/model_context_matrix.tsv
```

例如：

| model | K562 | Liver | Brain |
|---|---:|---:|---:|
| Distance | yes | yes | yes |
| rE2G | yes | yes | yes |
| pgBoost | yes | partial | yes |
| Borzoi | yes | yes | yes |
| AlphaGenome | yes | yes | yes |

必须在正式 benchmark 前冻结。

---

# 46. Training leakage registry

建立：

```text
config/model_training_registry.tsv
```

字段：

```text
model
training_dataset
training_celltype
training_label_type
benchmark_overlap
```

---

# 47. Leakage 类型

分别标记：

```text
PAIR_LABEL_SEEN
CELLTYPE_SEEN
ASSAY_TRACK_SEEN
DATASET_SEEN
NO_KNOWN_OVERLAP
UNKNOWN
```

不能简单 yes/no。

---

# 48. Primary performance

Main strict analysis：

```text
PAIR_LABEL_SEEN != TRUE
```

也就是说真正用 benchmark pair 训练过的方法不能用同一 pair 证明自己好。

---

# 49. rE2G / scE2G CRISPR

因为模型训练涉及 K562 CRISPR：

Main strict held-out：

优先使用：

```text
non-K562 held-out
```

全 5-cell-type：

放 secondary analysis。

---

# 50. pgBoost

pgBoost 本身使用 eQTL 训练，因此：

```text
pgBoost on its training-style eQTL
```

不得作为最强独立验证。

pgBoost Main independent evaluation：

```text
CRISPR
GWAS
```

eQTL 表现作为：

```text
in-domain
```

单独标记。

---

# 51. Common benchmark

生成：

```text
COMMON_ALL_MODELS
```

条件：

```text
所有 Main model 都 applicable
```

用于严格横向比较。

---

# 52. Full coverage benchmark

同时生成：

```text
FULL_AVAILABLE
```

每个模型在自己所有 applicable samples 上测试。

因此文章同时回答：

1. 公平交集谁最强？
2. 实际覆盖率谁最大？

---

# 53. Model coverage

每个模型必须报告：

\[
Coverage
=
\frac{scored\ applicable\ pairs}
{all\ applicable\ pairs}
\]

不能只报告 performance。

---

# 54. Main ranking metrics

主指标：

## MRR

\[
MRR=
\frac1N\sum_i\frac1{rank_i}
\]

---

## Top-1

\[
Top1=P(rank=1)
\]

---

## Recall@3

\[
P(rank\le3)
\]

---

## Recall@5

\[
P(rank\le5)
\]

---

## NDCG

用于 multi-target variant。

---

# 55. 多 target gene

一个 variant 可以：

```text
gene A
gene B
```

同时为 gold。

因此不能强行选一个。

使用：

```text
gold_gene_set
```

Primary rank：

```text
best gold rank
```

并额外：

```text
Recall all gold genes
```

放 Supplementary。

---

# 56. Classification metrics

在存在 confident negative 时计算：

```text
AUPRC
AUROC
MCC
```

Main classification metric：

```text
AUPRC
```

因为正负极度不平衡。

---

# 57. Direction benchmark

有 signed effect 的数据：

真实：

```text
beta > 0
beta < 0
```

模型：

```text
signed score
```

评价：

```text
direction accuracy
balanced accuracy
MCC
Spearman
```

---

# 58. Effect-size benchmark

真实：

```text
eQTL beta
experimental Δexpression
```

模型：

```text
predicted Δ expression
```

评价：

```text
Pearson
Spearman
R²
```

---

# 59. Distance stratification

固定：

```text
0–10 kb
10–50 kb
50–100 kb
100–250 kb
250–500 kb
500 kb–1 Mb
```

---

# 60. Nearest-gene stratification

固定：

```text
gold nearest
gold second nearest
gold third nearest
gold rank >=4
```

必须独立画图。

---

# 61. PIP stratification

```text
0.50–0.70
0.70–0.90
0.90–0.95
>=0.95
```

---

# 62. Context stratification

```text
exact matched
closely matched
coarse matched
unmatched
```

Primary：

```text
exact + closely matched
```

---

# 63. Evidence stratification

```text
CRISPR
eQTL
GWAS
curated disease
```

绝不能把它们全混一个 AUPRC。

---

# 64. Bootstrap

所有 confidence intervals：

```text
2,000 bootstrap replicates
```

抽样单位不能是 pair。

V2G：

```text
variant/locus
```

CRISPR：

```text
enhancer / experiment cluster
```

---

# 65. Paired model comparison

比较模型 A、B：

每次 bootstrap：

抽完全相同 loci。

计算：

\[
\Delta MRR
=
MRR_A-MRR_B
\]

报告：

```text
mean Δ
95% CI
bootstrap P
```

---

# 66. Multiple testing

所有 pairwise method comparisons：

```text
Benjamini-Hochberg FDR
```

固定：

```text
q < 0.05
```

---

# 67. Random baseline

Random model：

每个 variant：

candidate genes 按：

```text
SHA256(variant_id + gene_id + seed)
```

稳定排序。

seed：

```text
20260904
```

不得使用不同随机结果。

---

# 68. Integrated model

只有 benchmark 完成后自动训练。

不允许人为决定特征。

固定：

```text
log_distance
ABC
rE2G
scE2G
pgBoost
Borzoi_abs
AlphaGenome_abs
Enformer_abs
```

不存在的 model score：

增加 missing indicator。

---

# 69. Integrated Model 1：rank mean

无训练。

对每个 variant：

模型 score 转 percentile rank。

\[
S=
mean(rank\ percentile)
\]

---

# 70. Integrated Model 2：logistic regression

标准化 feature。

模型：

```text
L2 logistic regression
```

固定：

```text
class_weight=balanced
C grid predefined
```

使用 nested CV。

---

# 71. Integrated Model 3：XGBoost

固定 outer CV：

**按 chromosome。**

禁止 random row split。

Open Targets 原 L2G 本身也使用 chromosome-separated outer CV 来避免 locus leakage。

---

# 72. Chromosome folds

固定：

```text
Fold1: chr1 chr2 chr3 chr4
Fold2: chr5 chr6 chr7 chr8
Fold3: chr9 chr10 chr11 chr12
Fold4: chr13 chr14 chr15 chr16
Fold5: chr17 chr18 chr19 chr20 chr21 chr22
```

---

# 73. Ensemble 禁止数据泄漏

任何 fold：

```text
test chromosome
```

不得参与：

```text
scaler
feature selection
hyperparameter tuning
threshold selection
```

---

# 74. Preflight

在任何正式下载前运行：

```text
scripts/preflight/preflight.py
```

检查：

```text
OS
Python
mamba
Snakemake
bcftools
bedtools
samtools
git
wget/curl
aria2c
SLURM
GPU
CUDA
disk
internet
GTEx
EBI
GitHub
Zenodo
Synapse
AlphaGenome
```

---

# 75. Preflight disk

推荐：

```text
scratch >= 1 TB free
persistent >= 300 GB free
```

如果：

```text
scratch < 800 GB
```

直接 FAIL。

原因之一是 ENCODE-rE2G comparison prediction bundle 本身即可达到几十 GB。

---

# 76. GPU

推荐：

```text
A100 40 GB+
```

最低允许：

```text
24 GB
```

低显存通过自动 batch-size 调整处理。

---

# 77. AlphaGenome API smoke test

正式运行前：

随机固定测试 variant。

必须成功：

```text
score_variant
RNA_SEQ scorer
gene output
```

否则整个 pipeline 不启动。

---

# 78. Borzoi smoke test

必须：

1. 权重全部下载；
2. 4 replicate 全部能加载；
3. chr22 test variant 可推理；
4. REF/ALT score finite。

否则 pipeline 不启动。

---

# 79. Enformer smoke test

同理：

```text
weight load
REF inference
ALT inference
human output
CAGE tracks
```

必须全部 PASS。

---

# 80. Published bundle smoke test

至少读取：

```text
5 rows
```

确认：

```text
coordinate
gene
score
```

存在。

否则不能开始。

---

# 81. Resource lock

preflight 成功后生成：

```text
data/locked/provenance.lock.yaml
```

记录：

```text
download URL
Git commit
release
timestamp
SHA256
file size
software version
model version
```

正式运行之后禁止改变。

---

# 82. Git repository 固定方式

不能：

```bash
git clone master
```

然后算完以后不知道版本。

正确：

```text
clone
resolve HEAD commit
write commit SHA into lock
checkout detached commit
```

以后重跑：

使用同一 commit。

---

# 83. Download strategy

所有大型下载：

```text
aria2c
```

要求：

```text
resume
multiple connection
checksum
retry
```

不得简单 Python `requests.get()` 下载 70GB。

---

# 84. 下载失败

规则：

```text
retry 5
resume partial download
verify checksum
```

失败 5 次：

```text
FATAL_PRECHECK_RESOURCE
```

正式 workflow 不开始。

---

# 85. 正式运行中任务失败策略

Snakemake：

```text
retries: 3
```

远程 API：

更高层 exponential retry。

GPU OOM：

自动降低 batch。

---

# 86. 不允许 silent failure

任何失败必须写：

```text
results/failures/failures.parquet
```

字段：

```text
stage
dataset
model
shard
error_type
message
retry_count
resolved
```

---

# 87. “全部完成”的严格定义

最终：

```text
unresolved fatal errors = 0
```

否则：

```text
run_status = INCOMPLETE
exit code != 0
```

不能生成一个看起来成功的论文结果。

---

# 88. Snakemake rule DAG

核心：

```text
preflight
  ↓
lock_resources
  ↓
download_references
download_gold_standards
download_published_predictions
download_models
  ↓
normalize_variants
normalize_genes
normalize_contexts
  ↓
build_gold_registry
build_candidate_sets
build_leakage_registry
build_applicability_matrix
  ↓
score_baselines
score_published_models
score_borzoi
score_enformer
score_alphagenome
  ↓
merge_predictions
  ↓
evaluate_ranking
evaluate_classification
evaluate_direction
evaluate_effect_size
  ↓
bootstrap
stratify
failure_analysis
  ↓
train_integrated_models
evaluate_integrated_models
  ↓
make_figures
make_tables
  ↓
final_qc
  ↓
release
```

---

# 89. `run_once.sh`

最终只能包含类似：

```bash
#!/usr/bin/env bash
set -euo pipefail

source .env

python scripts/preflight/preflight.py \
    --config config/site.yaml

snakemake \
    --profile config/slurm \
    --use-conda \
    --rerun-incomplete \
    --keep-going \
    --printshellcmds \
    --show-failed-logs \
    all

python scripts/preflight/final_qc.py
```

用户只运行这一条。

---

# 90. Slurm

建立：

```text
config/slurm/config.yaml
```

CPU small：

```text
mem 16G
cpus 4
```

large table：

```text
mem 64G
cpus 16
```

GPU sequence：

```text
gpu 1
mem 64G
cpus 8
```

大 merge：

```text
mem 128G
```

---

# 91. 数据处理工具

大表：

```text
Polars
DuckDB
PyArrow
```

禁止：

```text
pandas 一次读 100GB
```

Pandas 只用于小表和最终 plotting inputs。

---

# 92. 核心 Python package

所有逻辑放：

```text
src/v2gbench/
```

不能全塞：

```text
scripts/*.py
```

scripts 只负责 CLI。

---

# 93. Schema validation

建议使用：

```text
Pandera 或 Pydantic
```

每一步输入输出均 validate。

---

# 94. 必须写的 unit tests

至少：

```text
test_variant_normalization
test_ref_match
test_gene_mapping
test_context_mapping
test_gold_in_candidate_set
test_no_duplicate_candidate
test_score_direction
test_ranking
test_multitarget
test_missing_score_policy
test_context_applicability
test_training_overlap
test_bootstrap
test_deterministic_sampling
test_alphagenome_parser
test_borzoi_parser
test_enformer_parser
```

---

# 95. 最关键 test

必须：

```python
assert gold_gene in candidate_gene_set
```

对 100% benchmark instances 成立。

否则立即 FATAL。

---

# 96. Reference QC

要求：

```text
variant REF agreement >=99.9%
```

非 PASS variants：

全部写报告。

---

# 97. Gene QC

至少：

```text
>=99% gene IDs successfully resolved
```

未解决 gene：

不能悄悄扔掉。

---

# 98. Context QC

Primary benchmark：

至少：

```text
95% rows have context mapping
```

如果不足：

仍生成报告，但 Main context-specific analysis必须标识 coverage。

---

# 99. Model score QC

每个 model：

```text
NaN fraction
Inf fraction
score range
number unique values
coverage
```

全部输出：

```text
results/tables/model_score_qc.tsv
```

---

# 100. Sequence model sanity

检查：

```text
REF == ALT synthetic variant
```

score 应接近：

```text
0
```

---

# 101. Allele swap sanity

随机选一小组：

```text
REF→ALT
ALT→REF
```

signed score 应近似：

\[
S_{reverse}\approx -S_{forward}
\]

明显不成立：

说明 allele implementation 有 bug。

---

# 102. Pipeline reproduction sanity checks

正式 benchmark 前必须复现至少：

### CRISPR

published ABC / rE2G 大致 ranking。

### pgBoost

作者公开 evaluation set 上：

模型相对表现与 published metrics 相符。

### TraitGym

直接读取作者 prediction 后重算 metric：

与其公开 metric 接近。

### Borzoi

官方 QTL sample benchmark 小规模复现。

---

# 103. Reproduction tolerance

不能要求 floating-point 完全一样。

预设：

```text
metric absolute difference <= 0.02
```

若 >0.02：

```text
SANITY_FAIL
```

正式 Main analysis 不启动。

---

# 104. Main Figure 1

标题：

> A unified benchmark of regulatory variant-to-gene prediction

Panel：

### 1A

benchmark schematic。

### 1B

数据来源数量：

```text
variants
elements
genes
contexts
positive pairs
tested negatives
```

### 1C

UpSet：

不同 gold-standard overlap。

### 1D

gold target distance。

### 1E

nearest/non-nearest proportion。

---

# 105. Figure 2

> Existing models show highly heterogeneous performance across regulatory linking tasks

Heatmap：

```text
model
×
benchmark
```

metrics：

```text
MRR
Top1
AUPRC
coverage
```

---

# 106. Figure 3

> Distal and non-nearest regulatory interactions remain challenging

至少：

### 3A

MRR vs distance。

### 3B

nearest vs non-nearest。

### 3C

gold distance rank：

```text
1
2
3
4+
```

### 3D

candidate gene number。

---

# 107. Figure 4

> Cell-type matching improves regulatory target prediction

比较：

```text
exact context
related context
coarse context
```

---

# 108. Figure 5

> Sequence and enhancer-gene models capture complementary information

例如：

X：

```text
AlphaGenome/Borzoi score
```

Y：

```text
rE2G/ABC score
```

展示：

```text
both correct
sequence only
E2G only
both fail
```

---

# 109. Figure 6

> Integrating sequence variant effects and enhancer-gene evidence improves V2G prioritization

比较：

```text
best standalone
rank-average
logistic
XGBoost
```

在 chromosome-held-out 上报告。

---

# 110. Main Results Table

```text
Model
Family
MRR
Top1
Top3
Top5
AUPRC
Coverage
95% CI
```

---

# 111. Supplementary Tables

自动生成：

```text
S1 Dataset registry
S2 Gold standard summary
S3 Context mapping
S4 Dataset overlaps
S5 Model registry
S6 Model versions
S7 Model applicability
S8 Training leakage
S9 All model configurations
S10 Overall metrics
S11 Stratified metrics
S12 Pairwise bootstrap comparisons
S13 Failure-mode loci
S14 Sequence model scores
S15 Integrated model feature importance
S16 QC report
S17 Excluded models and reasons
```

---

# 112. Supplementary sensitivity

全部提前冻结：

```text
PIP 0.50
PIP 0.70
PIP 0.90
PIP 0.95

window 250kb
window 500kb
window 1Mb

all GENCODE
context-tested genes

missing=0
common coverage

all contexts
high-confidence contexts

all evidence
strict independent evidence
```

不得后来挑对自己有利的组合。

---

# 113. Failure analysis

建立：

```text
results/failures/error_cases.parquet
```

字段：

```text
variant
context
gold_gene
gold_rank
nearest_gene
best_wrong_gene
distance
model
score_gold
score_wrong
evidence
```

---

# 114. Automatic interesting locus selection

不允许手工挑 case。

固定规则：

Case A：

```text
Distance wrong
>=3 advanced models correct
gold non-nearest
```

Case B：

```text
sequence correct
all E2G wrong
```

Case C：

```text
E2G correct
all sequence wrong
```

Case D：

```text
all models wrong
high confidence gold
```

每类自动选：

```text
top 5
```

用于论文 discussion。

---

# 115. Model disagreement analysis

计算 model rank correlation：

```text
Spearman
```

以及：

```text
prediction overlap
Jaccard top1
Jaccard top3
```

形成 method clustering。

---

# 116. Variant-effect vs V2G

利用 TraitGym / sequence score：

建立：

```text
variant_effect_performance
```

对比：

```text
V2G performance
```

回答：

> 一个很强的 regulatory variant model 是否必然会找对 gene？

这是文章一个重要 conceptual result。

---

# 117. 不纳入 Main V2G 的模型

例如：

```text
CADD
phyloP
phastCons
GPN
Sei
DeepSEA
SVEN
PromoterAI
```

如果模型本身没有 target-gene-specific score：

不作为 standalone V2G model。

但可记：

```text
variant_effect_only
```

SVEN 等模型能预测 tissue-specific regulatory impact，但本质仍主要是 variant regulatory-effect scorer，而非明确 target-gene linker。

PromoterAI 本身针对 promoter variant，也不属于 enhancer-V2G 主任务。

---

# 118. GET

GET 是重要 transcription foundation model，但其输入和任务设计不是 nucleotide-level enhancer-variant → target-gene zero-shot score，因此不直接放 Main V2G leaderboard。GET 官方目前主要以 cell-type chromatin accessibility 等信息预测 gene expression。

记录：

```text
excluded_from_main_reason =
no_direct_zero_shot_nucleotide_V2G_score
```

而不是假装没看到它。

---

# 119. DNALongBench

DNALongBench 包含：

```text
Enhancer-target gene
eQTL
```

long-range sequence tasks，可作为 Supplementary external benchmark。

但是它属于：

```text
task-specific foundation model benchmark
```

不是本文统一 biological V2G gold standard。

因此自动下载其 published test predictions / metrics，放：

```text
Supplementary External Benchmark
```

不混 Main。

---

# 120. 一次性资源发现

创建：

```text
scripts/preflight/discover_resources.py
```

自动完成：

```text
GitHub release discovery
Zenodo record discovery
Synapse manifest inspection
GTEx file existence
eQTL Catalogue metadata
AlphaGenome package/API
```

并生成：

```text
resource_manifest.lock.tsv
```

后续所有 Snakemake 只读 lock。

不得再次动态查最新版本。

---

# 121. 不能“自动更新到最新版”

这是 reproducibility 大忌。

第一次 run：

锁定今天能访问到的版本。

后面：

全部使用：

```text
provenance.lock.yaml
```

---

# 122. 完整配置 `project.yaml`

必须至少：

```yaml
project:
  name: V2G-Benchmark-OneShot
  version: "1.0"
  genome_build: GRCh38
  gencode: v47
  seed: 20260904

benchmark:
  primary_window: 1000000
  sensitivity_windows:
    - 250000
    - 500000
    - 1000000

eqtl:
  primary_pip: 0.90
  sensitivity_pip:
    - 0.50
    - 0.70
    - 0.90
    - 0.95
  negative_pip: 0.01

sequence_core:
  max_variant_contexts: 5000

bootstrap:
  replicates: 2000

context:
  primary_min_confidence: 0.8

execution:
  retries: 3
  fail_on_unresolved: true
```

这个文件正式 run 后禁止改。

---

# 123. Model config

```yaml
models:

  random:
    family: baseline
    mode: derived

  nearest_tss:
    family: baseline
    mode: derived

  abc:
    family: e2g
    mode: published_prediction

  encode_re2g:
    family: e2g
    mode: published_prediction
    paper_version: "1.0.0"

  sce2g_atac:
    family: e2g
    mode: published_prediction

  sce2g_multiome:
    family: e2g
    mode: published_prediction

  epimap:
    family: e2g
    mode: published_prediction

  graphreg:
    family: e2g
    mode: published_prediction

  pgboost:
    family: singlecell
    mode: published_prediction

  scent:
    family: singlecell
    mode: published_prediction

  signac:
    family: singlecell
    mode: published_prediction

  archr:
    family: singlecell
    mode: published_prediction

  cicero:
    family: singlecell
    mode: published_prediction

  borzoi:
    family: sequence
    mode: local_inference
    ensemble_replicates: 4

  enformer:
    family: sequence
    mode: local_inference

  alphagenome:
    family: sequence
    mode: remote_inference

  opentargets_l2g:
    family: disease
    mode: published_prediction
```

---

# 124. Mandatory model rule

所有上面：

```text
enabled: true
```

的模型：

preflight 必须验证：

```text
RESOURCE_AVAILABLE
```

否则停止。

不能：

> GraphReg 下载失败，那先不做吧。

---

# 125. Dynamic model rule

ENCODE bundle 中自动发现的额外 method：

自动：

```text
enabled = supplementary
```

只要 prediction schema 有效：

全部跑。

---

# 126. Data adapter interface

每个 dataset adapter 必须实现：

```python
class DatasetAdapter:

    def download(self): ...
    def validate_raw(self): ...
    def harmonize(self): ...
    def qc(self): ...
    def provenance(self): ...
```

不能每个脚本随便写。

---

# 127. Model adapter interface

每个模型：

```python
class ModelAdapter:

    def validate_resources(self): ...
    def applicability(self, context): ...
    def score(self, inputs): ...
    def normalize_score(self): ...
    def qc(self): ...
```

---

# 128. Evaluation code禁止知道模型名字

错误：

```python
if model == "AlphaGenome":
```

Evaluation 层不得出现。

它只能读取：

```text
model_id
ranking_score
```

所有特殊逻辑只能在 model adapter。

---

# 129. Benchmark code禁止知道数据源特殊列名

特殊列名映射只允许：

```text
DatasetAdapter
```

完成。

核心 benchmark 层只使用 canonical schema。

---

# 130. 日志

每个 job：

```text
logs/{rule}/{wildcard}.out
logs/{rule}/{wildcard}.err
```

---

# 131. Run metadata

生成：

```text
results/release/run_manifest.json
```

包括：

```text
run_id
start
end
hostname
Slurm cluster
git commit
data lock hash
config hash
number jobs
failed jobs
resolved retries
```

---

# 132. Config hash

run 开始：

计算：

```text
SHA256(config/*)
```

如果中途 config 被修改：

final QC 必须失败。

保证：

> 跑到一半不能偷偷改 PIP threshold。

---

# 133. Code hash

同理记录：

```text
Git commit
dirty status
```

正式 run 开始时：

```text
git status --porcelain
```

必须为空。

否则：

```text
PRECHECK_FAIL_DIRTY_REPO
```

---

# 134. 正式运行禁止修改代码

`run_once.sh` 开始后生成：

```text
RUN_LOCK
```

记录 git tree hash。

final QC 再计算。

不同：

FAIL。

---

# 135. 最终 QC

必须执行：

```text
scripts/preflight/final_qc.py
```

检查：

```text
all mandatory datasets PASS
all mandatory models PASS
all model shards PASS
no unresolved fatal errors
all figures exist
all tables exist
all metrics finite
gold candidate coverage 100%
config hash unchanged
git tree unchanged
```

---

# 136. SUCCESS 文件

只有 final QC 全部通过：

创建：

```text
results/release/SUCCESS
```

内容：

```text
V2G-Benchmark-OneShot completed successfully.
```

没有这个文件：

就不能称为跑完。

---

# 137. 最终 release

自动生成：

```text
results/release/
│
├── SUCCESS
├── README.txt
├── run_manifest.json
├── provenance.lock.yaml
├── software_versions.tsv
├── data_checksums.tsv
├── model_registry.tsv
├── dataset_registry.tsv
├── benchmark_registry.parquet
├── model_predictions.parquet
├── metrics.tsv
├── SupplementaryTables.xlsx
├── figures/
└── logs_summary/
```

---

# 138. README 最终自动填写

不得手动填写最终数字。

README 中数据规模：

```text
N variants
N enhancer-gene pairs
N genes
N contexts
N models
```

全部由脚本读实际结果自动填。

---

# 139. 论文主要预期结论不得写死

代码和 Figure 不能假设：

```text
AlphaGenome 最好
```

也不能假设：

```text
integrated model 一定最好
```

如果：

```text
Nearest TSS 最好
```

也必须如实输出。

---

# 140. 建议文章题目

主标题可以冻结为工作标题：

> **From Variant to Gene: A unified benchmark of regulatory target-gene prediction**

备选：

> **Benchmarking computational models for linking noncoding regulatory variants to target genes**

---

# 141. 文章核心贡献定义

论文不能只宣传：

> 我跑了很多模型。

真正 contribution：

### Contribution 1

统一多种已有 gold-standard benchmark。

### Contribution 2

统一：

```text
variant
gene
context
candidate universe
```

### Contribution 3

严格区分：

```text
variant effect
E2G
V2G
L2G
```

### Contribution 4

系统处理：

```text
training leakage
context matching
model coverage
distal regulation
non-nearest genes
```

### Contribution 5

比较：

```text
sequence models
E2G models
single-cell methods
classical baselines
```

### Contribution 6

测试组合模型是否提升 V2G。

---

# 142. Codex 必须遵守的执行顺序

Codex 不能：

> 写一部分先跑，之后再问用户。

必须：

### Phase 1

一次性创建完整 repository。

### Phase 2

全部 tests 写完。

### Phase 3

全部 download adapters 写完。

### Phase 4

全部 dataset adapters 写完。

### Phase 5

全部 model adapters 写完。

### Phase 6

全部 benchmark/evaluation 写完。

### Phase 7

全部 plotting 写完。

### Phase 8

运行 unit tests。

### Phase 9

运行 dry run。

### Phase 10

运行 smoke test。

### Phase 11

preflight。

### Phase 12

一次性正式执行：

```bash
bash run_once.sh
```

禁止进入：

```text
“先跑 GTEx 看看”
```

模式。

---

# 143. Codex 不得询问的问题

如果规范已经定义：

不得再问：

```text
PIP 用 0.9 还是 0.95？
candidate window 多大？
missing prediction 怎么办？
用哪个 GENCODE？
多 target 怎么算？
模型 score 绝对值还是 signed？
```

答案已经在本规范。

---

# 144. Codex 遇到未知原始列名

正确做法：

```text
inspect schema
write deterministic adapter
write unit test
```

错误做法：

```text
让用户手动告诉它列名
```

---

# 145. 上游格式变化

adapter 必须：

1. 检测 expected aliases；
2. 根据 schema 而不是 column position；
3. 如果无法识别：

```text
FATAL_SCHEMA_CHANGE
```

不得猜。

---

# 146. 一次执行的最重要保障

项目在正式运行前必须实现：

```text
pytest
↓
snakemake --dry-run
↓
smoke test
↓
preflight
↓
FULL RUN
```

而不是：

```text
FULL RUN
↓
遇错
↓
修
↓
再跑
```

---

# 147. Smoke dataset

自动建立：

```text
tests/data/smoke/
```

固定：

```text
chr22
20 variants
~100 candidate genes/pairs
几个 CRISPR pairs
```

所有模型先跑 smoke。

---

# 148. AlphaGenome smoke

建议只：

```text
2 variants
```

验证 API。

---

# 149. Borzoi smoke

4 replicate 均跑：

```text
2 variants
```

---

# 150. Enformer smoke

```text
2 variants
```

---

# 151. Published bundle smoke

每 model：

```text
100 prediction rows
```

完成 parser test。

---

# 152. Model reproducibility

published prediction 模型：

记录：

```text
source file checksum
```

Sequence model：

记录：

```text
weights checksum
```

AlphaGenome：

记录：

```text
API package
model version returned by service
scorer configuration
```

---

# 153. 最重要的三个最终数据表

## 1. Gold registry

```text
benchmark_registry.parquet
```

---

## 2. Candidate V2G table

```text
candidate_v2g.parquet
```

---

## 3. Prediction table

```text
all_model_predictions.parquet
```

后续所有分析只能读这三类 canonical 表。

---

# 154. `all_model_predictions.parquet`

最终类似：

| variant | gene | gold | model | score | rank |
|---|---|---:|---|---:|---:|
| v1 | G1 | 0 | AlphaGenome | .03 | 3 |
| v1 | G2 | 1 | AlphaGenome | .41 | 1 |
| v1 | G3 | 0 | AlphaGenome | .12 | 2 |
| v1 | G1 | 0 | rE2G | .20 | 2 |
| v1 | G2 | 1 | rE2G | .81 | 1 |

整个项目最终就是系统评价：

> 对每个 variant，哪个模型最容易把 gold gene 排到第一？

---

# 155. 项目最终完成标准

以下 15 条必须全部满足：

```text
[ ] all mandatory datasets downloaded
[ ] all checksums recorded
[ ] all variants normalized
[ ] all REF alleles validated
[ ] all genes harmonized
[ ] all contexts harmonized
[ ] all gold genes in candidate sets
[ ] all mandatory models executed/imported
[ ] AlphaGenome completed
[ ] Borzoi 4-replicate completed
[ ] Enformer completed
[ ] published comparison models imported
[ ] all metrics generated
[ ] all figures generated
[ ] SUCCESS file created
```

少一项：

项目状态：

```text
INCOMPLETE
```

---

# 156. 最终推荐主榜单

为了让论文不失控，Main manuscript 固定比较：

```text
Random
Nearest-TSS
Distance-exp100k
ABC
ENCODE-rE2G
scE2G
EpiMap
GraphReg
pgBoost
SCENT
Signac
ArchR
Cicero
Enformer
Borzoi
AlphaGenome
Integrated-Rank
Integrated-Logistic
Integrated-XGBoost
```

其它自动发现 configuration：

全部进入 Supplement。

---

# 157. 本项目最终科学问题

最终不是问：

> AlphaGenome AUC 是多少？

而是系统回答：

### Q1

现有模型到底能不能：

\[
variant\rightarrow correct\ gene
\]

### Q2

当 target 是 nearest gene 时表现如何？

### Q3

当 target 跳过 nearest gene 时表现如何？

### Q4

距离超过 100–500 kb 后表现如何？

### Q5

cell-type matched evidence 是否明显改善？

### Q6

sequence foundation model 是否真正超过 distance baseline？

### Q7

sequence model 是否优于专门的 E2G models？

### Q8

不同模型是否具有互补信息？

### Q9

VariantEffect × E2G 是否比任一单模型更好？

### Q10

当前 V2G prediction 最大 failure mode 是什么？

---

# 158. Agent 的最终任务定义

Agent 的工作不是：

> 写一些示例代码。

而是：

> **完整实现、测试并执行本规范描述的 V2G-Benchmark-OneShot v1.0 工程。**

Agent 必须持续工作直到：

```text
results/release/SUCCESS
```

生成。

不得在中途因为：

```text
代码很多
下载很多
模型很多
运行时间长
```

而自行减少任务范围。

不得未经规范允许：

```text
删除模型
减少 benchmark
降低 gold standard 数量
跳过 sequence model
跳过 bootstrap
跳过 QC
```

---

# 159. 项目完成后的最低论文故事

如果 integrated model 没提升：

文章仍然成立：

> **Systematic benchmarking reveals limitations and context dependence of regulatory variant-to-gene prediction.**

如果 integrated model 提升：

文章升级为：

> **Sequence variant effects and enhancer–gene maps provide complementary information for regulatory variant-to-gene prediction.**

因此项目本身不依赖：

```text
一定训练出更好的模型
```

才有价值。

---

# 160. 最终工程哲学

整个项目必须遵守一句话：

> **Import what authors have already computed; recompute only what must be recomputed; harmonize everything; evaluate everything under one frozen framework.**

也就是：

```text
别人已经整理好的 Gold Standard
        +
别人已经算好的 published predictions
        +
真正需要自己跑的 AlphaGenome / Borzoi / Enformer
        +
统一 candidate genes
        +
统一 ranking metrics
        +
严格 leakage / context / coverage 控制
        =
一次性完整 V2G benchmark
```

这是整个工程的最终冻结设计。