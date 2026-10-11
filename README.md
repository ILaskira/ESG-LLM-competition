# 📝 比賽技術細節報告 | ESG Commitment & Evidence Analysis

**🏆 最終排名：65 / 344 Teams（Top 19%）**

**核心技術：NLP、MacBERT、Transformer、Multi-Task Learning、PyTorch、Auxiliary Supervision**

> 🔒 **資料保密聲明（Data Confidentiality）**
>
> 本專案遵守競賽資料使用與保密原則，僅公開競賽任務、模型方法論、通用技術架構及競賽成果。不公開原始資料、實際使用的欄位與特徵、資料分布、標註內容或其他可能涉及競賽限制的資訊。

---

## 📚 內容導覽

- [1. 賽題簡介](#1-賽題簡介)
- [2. 競賽任務與評分機制](#2-競賽任務與評分機制)
- [3. 整體建模策略](#3-整體建模策略)
- [4. 多任務深度學習架構](#4-多任務深度學習架構)
- [5. 輔助式監督學習](#5-輔助式監督學習)
- [6. 模型訓練與優化策略](#6-模型訓練與優化策略)
- [7. 模型評估與預測策略](#7-模型評估與預測策略)
- [8. 模型建構上的挑戰與策略](#8-模型建構上的挑戰與策略)
- [9. 結論與我的貢獻](#9-結論與我的貢獻)
- [10. 最終成績與排名](#10-最終成績與排名)

---

## 1. 賽題簡介

本次競賽以企業 **ESG（Environmental, Social, Governance）永續報告書** 為分析對象，透過自然語言處理（Natural Language Processing, NLP），辨識企業提出的永續承諾，以及相關支持證據的完整性與清晰度。

隨著 ESG 資訊揭露的重要性日益提升，企業永續報告書不僅需要說明永續發展方向，也需要提供具體、可追蹤及可驗證的行動承諾。

然而，傳統人工審閱方式需要投入大量時間，且不同審閱者可能對承諾及證據的判斷標準存在差異。

因此，本競賽希望透過人工智慧技術，建立可自動化分析 ESG 承諾與佐證資訊的模型。

### 🎯 競賽核心目標

- **承諾辨識（Commitment Identification）**：判斷企業是否提出具體的未來行動承諾。
- **證據辨識（Evidence Identification）**：判斷承諾是否具有支持證據。
- **證據清晰度分析（Clarity Analysis）**：評估支持證據的語意清晰程度。
- **驗證時間推論（Timeline Prediction）**：推論承諾預期可被驗證的時間範圍。

### 📌 本專案研究方向

本專案以 **Transformer-Based Multi-Task Learning** 為主要建模方向，整合中文預訓練語言模型、多任務分類及輔助式監督學習，嘗試建立能同時處理多項 ESG 語意判斷任務的深度學習架構。

---

## 2. 競賽任務與評分機制

本競賽參考 ML-Promise 相關研究成果，採用多任務綜合評分機制。

不同於一般單一標籤分類問題，本競賽同時評估四個具有不同難度與實務意義的 NLP 任務。

### 📊 四項核心任務

| Task | Description | Weight |
| :--- | :--- | :---: |
| Commitment Classification | 判斷文本是否表達企業承諾 | 20% |
| Evidence Identification | 判斷承諾是否具備支持證據 | 30% |
| Clarity Classification | 評估證據語意清晰度 | 35% |
| Timeline Classification | 預測承諾驗證時間 | 15% |

### 📐 評分方式：Weighted Macro-F1

官方採用 Macro-F1 評估各項任務，再根據權重計算綜合分數。

`Final Score = 0.20 × F1_Commitment + 0.30 × F1_Evidence + 0.35 × F1_Clarity + 0.15 × F1_Timeline`

### 🔍 評分機制的技術意涵

**1. Macro-F1 對各類別給予相同重要性**

Macro-F1 會先計算各類別的 F1-score，再取平均。

因此，模型不能只依賴預測主要類別取得良好成績，也需要兼顧較難辨識的類別。

**2. 證據相關任務佔比較高**

Evidence Identification 與 Clarity Classification 合計佔總分 **65%**。

這表示競賽不僅重視是否辨識到企業承諾，也重視模型對支持證據的理解能力。

**3. 各項任務具有語意關聯**

例如，當文本不存在承諾時，部分後續判斷必須符合相應的標註邏輯。

因此，模型除了需要具備分類能力，也必須考慮不同任務之間的關聯性。

---

## 3. 整體建模策略

### 🧠 核心想法：Multi-Task Learning

面對四項具有語意關聯的任務，本專案採用 **Multi-Task Learning（多任務學習）** 作為主要架構。

相較於為每個任務分別建立獨立模型，多任務學習可以透過共享語言模型，讓不同任務共同學習文本的語意表示。

### ✅ 選擇多任務架構的原因

- **共享語意表示（Shared Representation）**：不同任務可能需要理解相似的文本語意。
- **降低重複建模成本**：透過單一預訓練 Encoder 支援多個預測任務。
- **利用任務關聯性（Task Relatedness）**：讓不同任務共同提供訓練訊號。
- **擴充性（Scalability）**：可依需求加入不同的 Classification Heads 或 Auxiliary Tasks。

### 🔍 模型研究方向

本專案實驗中文預訓練 Transformer 模型，並進一步採用 MacBERT 建立多任務架構。

其中，MacBERT 為針對中文語言理解設計的預訓練模型，能提供具有上下文資訊的文字表示。

整體設計並非單純使用預訓練模型直接輸出分類結果，而是結合：

- Shared Transformer Encoder
- Multiple Classification Heads
- Multi-Task Loss
- Auxiliary Supervision
- Class Imbalance Handling
- Rule-Based Post-Processing

形成完整的 NLP 建模流程。

---

## 4. 多任務深度學習架構

### 🏗️ Model Architecture

本專案以 **MacBERT + Multi-Task Classification Heads** 作為核心架構。

```text
                 Input Text
                      │
                      ▼
              MacBERT Tokenizer
                      │
                      ▼
             Shared MacBERT Encoder
                      │
          Shared Text Representation
                      │
       ┌──────────────┼──────────────┐
       │              │              │
       ▼              ▼              ▼
 Commitment       Evidence       Clarity
Classification  Classification  Classification
       │              │              │
       └──────────────┼──────────────┐
                      │              │
                      ▼              ▼
                Timeline        Auxiliary
              Classification     Tasks
                      │              │
                      └──────┬───────┘
                             ▼
                       Multi-Task
                      Optimization
```

### 🔹 Shared Encoder

所有任務共享同一個 MacBERT Encoder。

模型首先將輸入文字轉換為上下文語意表示，再交由不同的預測模組進行分類。

### 🔹 Task-Specific Heads

針對四項官方任務，分別建立獨立的 Classification Head。

這種設計使模型可以在共享底層語意資訊的同時，針對不同任務學習專屬的分類邊界。

### 💡 架構上的主要考量

共享 Encoder 可以避免完全重複的語言表示學習，但多任務模型也可能出現 **Negative Transfer（負遷移）**。

例如，不同任務的學習難度、收斂速度及損失尺度可能不同，進而產生訓練衝突。

因此，本專案除了設計多任務架構，也透過任務權重與輔助學習機制調整模型的訓練目標。

---

## 5. 輔助式監督學習

### 🧩 為什麼需要 Auxiliary Supervision？

傳統文本分類通常透過句子或段落的整體表示預測分類結果。

然而，ESG 承諾與證據分析需要的不只是整體分類，也包含對文本中關鍵語意資訊的理解。

因此，本專案在四項主任務以外，加入 **Auxiliary Supervision（輔助式監督學習）**，提供額外的學習訊號。

### 🔹 方法一：Span Detection

Span Detection 是 NLP 常見的資訊定位方法，目標是辨識一段文字中特定資訊的起始與結束位置。

本專案將 Span Prediction 作為輔助任務，讓模型除了學習分類結果，也能學習與主要任務相關的局部文字表示。

**技術概念：**

`Input → Transformer Representation → Start / End Prediction`

### 🔹 方法二：Token-Level Classification

Token-Level Classification 則是針對文字中的各個 Token 建立分類任務。

與整段文本的 Classification 不同，Token-Level Learning 提供更細粒度的監督訊號。

**技術概念：**

`Token Representations → Token-Level Prediction`

### ✅ 輔助式學習的設計目的

- 增加共享 Encoder 接收到的監督訊號。
- 同時學習整體與局部語意表示。
- 讓模型不只依賴單一文本表示完成分類。
- 探索細粒度資訊學習對多任務分類的輔助效果。

> 💡 核心想法：讓模型同時學習「整體文本的判斷」與「局部語意資訊的表示」，而不只依賴單一分類目標。

---

## 6. 模型訓練與優化策略

### ⚙️ 多任務損失函數（Multi-Task Loss）

為了同時訓練不同的分類任務，本專案將多項 Loss 整合為共同的優化目標。

概念上可以表示為：

`Total Loss = Weighted Classification Loss + Auxiliary Loss`

其中：

- **Classification Loss**：負責官方四項分類任務。
- **Auxiliary Loss**：負責額外的局部語意監督學習。

此方法讓 Shared Encoder 在同一次訓練過程中，接收不同任務的梯度訊號。

### 🔹 Class-Weighted Cross-Entropy

由於 Macro-F1 對所有類別給予相同權重，模型需要避免過度偏向常見類別。

因此，本專案採用 Class-Weighted Cross-Entropy，調整不同類別在訓練過程中的 Loss 貢獻。

其主要目的是提升模型對少數類別的關注，而不是單純最大化整體 Accuracy。

### 🔹 任務權重設計（Task Weighting）

不同任務可能具有不同的收斂特性。

透過 Multi-Task Loss Weighting，可以調整各分類任務在整體訓練目標中的相對重要性。

需要注意的是，訓練 Loss 權重不一定需要與競賽官方評分權重完全相同。

### 🔹 模型選擇（Model Selection）

本專案以官方 Weighted Macro-F1 作為模型評估的重要依據，並建立最佳模型保存機制。

相較於只根據 Training Loss 選擇模型，此方法更貼近競賽實際評分目標。

### 🛠️ 使用技術

| Category | Technology |
| :--- | :--- |
| Programming | Python |
| Deep Learning Framework | PyTorch |
| Pretrained Language Model | MacBERT |
| NLP Framework | Hugging Face Transformers |
| Training Architecture | Multi-Task Learning |
| Auxiliary Learning | Span Detection, Token-Level Classification |
| Optimization | Weighted Cross-Entropy, Multi-Task Loss |
| Evaluation | Macro-F1, Weighted Macro-F1 |

---

## 7. 模型評估與預測策略

### 📊 以官方評分為核心

本專案將官方四項任務的 Weighted Macro-F1 作為模型評估基準。

與一般單任務模型不同，四項任務的表現會共同影響最終成績。

因此，優化策略不僅關注單一任務的最高分數，也需要考慮整體模型的綜合表現。

### 🔹 預測一致性（Prediction Consistency）

四項分類任務之間存在邏輯關聯，但多任務模型的不同 Classification Heads 仍可能產生不一致的結果。

例如，若模型判斷不存在明確承諾，其他任務的預測結果就需要符合官方對相關類別的定義。

因此，本專案設計 Rule-Based Post-Processing，讓模型輸出符合任務定義。

### 🔹 Rule-Based Post-Processing

模型完成預測後，可依照公開的競賽標註規則，進行必要的邏輯一致性檢查。

主要目的包括：

- 確保不同分類任務之間的結果一致。
- 避免不符合競賽類別定義的預測組合。
- 比較模型原始輸出與規則修正後的評估表現。

### 🔹 多版本預測策略比較

本專案亦設計不同後處理版本，透過驗證分數比較策略的有效性。

整體原則為：

**保留模型原始預測能力，同時利用明確的任務規則處理可能的邏輯衝突。**

---

## 8. 模型建構上的挑戰與策略

### 🧩 挑戰一：多項任務具有不同的學習難度

四項 ESG 任務的預測目標不同，模型需要在共享參數的條件下，學習各自的分類邊界。

**解決策略：Multi-Task Learning**

- 以 Shared Encoder 建立共同的語意表示。
- 透過 Task-Specific Heads 保留各任務的獨立預測能力。
- 使用 Multi-Task Loss 整合訓練目標。

### 🔍 挑戰二：整體文本分類可能忽略局部資訊

ESG 承諾與證據的判斷需要理解具體語意，單純的段落分類可能無法充分利用局部文字特徵。

**解決策略：Auxiliary Supervision**

- 增加 Span Detection 輔助任務。
- 增加 Token-Level Classification。
- 將句子層級與 Token 層級學習整合到同一架構。

### ⚖️ 挑戰三：Macro-F1 與類別不平衡

當模型過度偏向常見類別時，即使整體 Accuracy 不低，Macro-F1 仍可能受到影響。

**解決策略：Class-Weighted Loss**

- 調整不同類別在訓練中的損失貢獻。
- 以 Macro-F1 衡量分類模型。
- 根據官方綜合評分選取模型。

### 🔀 挑戰四：跨任務預測邏輯一致性

不同的 Classification Heads 分別產生預測結果，可能出現不符合任務定義的組合。

**解決策略：Rule-Based Post-Processing**

- 依照公開競賽規則檢查預測結果。
- 設計任務之間的一致性修正。
- 比較不同後處理版本的模型成效。

---

## 9. 結論與我的貢獻

本專案以 **Multi-Task MacBERT** 為主要研究架構，將多項 ESG 語意理解任務整合至統一的 Transformer 模型，並加入 Auxiliary Supervision 與預測後處理機制。

相較於單純使用預訓練語言模型進行分類，本專案更重視**多任務之間的資訊共享、訓練目標整合，以及預測結果的邏輯一致性**。

### 👨‍💻 我的技術貢獻 | Key Contributions

- **多任務模型設計（Multi-Task Model Design）**：建構以 MacBERT 為基礎的 Shared Encoder 架構，透過四個 Task-Specific Classification Heads，同時處理官方定義的四項 ESG 分類任務。
- **輔助式監督學習（Auxiliary Supervision）**：整合 Span Detection 與 Token-Level Classification，使模型同時學習整體文本分類及局部語意表示。
- **模型優化策略（Optimization Strategy）**：使用 Multi-Task Loss 與 Class-Weighted Cross-Entropy，處理多任務學習及類別不平衡所帶來的訓練挑戰。
- **評分導向模型選擇（Metric-Oriented Model Selection）**：依據官方 Weighted Macro-F1 設計模型評估與最佳權重保存機制，使模型選擇符合競賽的綜合評分目標。
- **預測邏輯一致性（Prediction Consistency）**：整合 Rule-Based Post-Processing，處理不同預測任務之間的標註邏輯關係。
- **最終競賽成果**：於 **344 支參賽隊伍中排名第 65 名（Top 19%）**，展現 NLP、多任務深度學習與模型優化的實作能力。

### 🎯 專案價值

本次競賽不僅是文字分類模型的應用，也是一個整合語言模型、多任務學習、細粒度監督及推論邏輯的實務案例。

相關技術具有延伸至金融文本分析、法規文件理解、企業資訊揭露分析與其他多任務 NLP 問題的潛力。

---

## 10. 最終成績與排名

| Item | Result |
| :--- | :--- |
| Competition | ESG Commitment & Evidence Analysis |
| Final Ranking | **65 / 344 Teams** |
| Ranking Percentile | **Top 19%** |
| Main Research Direction | Multi-Task NLP |
| Model Architecture | MacBERT + Multi-Task Learning |
| Main Technologies | PyTorch, Transformer, Auxiliary Supervision |

### 🏆 Competition Achievement

**Top 19% — 65th out of 344 Teams**

本次競賽透過中文預訓練語言模型與多任務深度學習架構，針對 ESG 承諾識別、證據判斷、證據清晰度及時間推論進行整合建模。

最終取得 **Top 19%** 的競賽成績，累積了 Transformer 架構設計、多任務模型訓練、NLP 輔助學習及評分導向優化的實務經驗。

---

## 🔒 Data Confidentiality

To comply with competition data restrictions, this repository documents only the publicly described tasks, general modeling methodology, high-level architecture, and competition achievements.

**No competition datasets, private feature specifications, sample records, annotation contents, or confidential implementation details are disclosed.**

---

**Keywords:** `ESG` · `NLP` · `MacBERT` · `Transformer` · `Multi-Task Learning` · `PyTorch` · `Auxiliary Supervision` · `Text Classification` · `Macro-F1`
