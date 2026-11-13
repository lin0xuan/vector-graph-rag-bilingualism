# MuSiQue 英文 20 题评测（2026-11-13）

## 口径

固定随机种子 20261113，从 MuSiQue-Ans v1.0 开发集抽取二跳 8 题、三跳 6 题、四跳 6 题。保留每题所有候选段落和干扰段落，386 段去重后建一个共享索引；查询使用上游原生 passage ID 过滤。初始实体/关系检索仍在共享索引上进行，再执行上游过滤；因此这不是每题独立 20 段索引，也不是官方排行榜配置。原始段落作为 passage，不再次切分。标准答案、问题分解和支持标签均未进入建库、检索或回答。

英文链路：固定 Zilliz 上游 commit `47590ad4eab5838cdcfec94f8a53ebfe70ebe330`；本地 BGE-large-en-v1.5 ONNX / 1024 维、Milvus Lite，原生图扩展、Jev 1.13.0 关系和段落两阶段评分、final_top_k=10。抽取和回答使用 Qwen/Qwen3.5-35B-A3B；语义评分使用 Qwen/Qwen3.6-35B-A3B。保留上游提示词；调用适配仅关闭 thinking、设置输出上限、检查完成状态和记录 token。

严格 EM/F1 使用随归档保存的 MuSiQue 官方评分源码，直接评价未经改写的上游答案，支持答案别名。上游常输出完整句子，因此严格 EM 不能单独反映事实正确率。参考答案语义匹配率是另一个模型的判断，允许同义实体名和完整句式；不等同于官方 EM。支持段落召回只表示找到对应段落，不保证每一步推理都正确。

## 结果

|分组|题数|参考答案语义匹配|原始 EM|原始 F1|支持段落找齐|平均支持召回|
|---|---:|---:|---:|---:|---:|---:|
|全部|20|10/20（50.0%）|0.0%|3.5%|16/20|92.1%|
|2 跳|8|4/8（50.0%）|0.0%|5.8%|7/8|93.8%|
|3 跳|6|2/6（33.3%）|0.0%|1.3%|5/6|94.4%|
|4 跳|6|4/6（66.7%）|0.0%|2.5%|4/6|87.5%|

## 费用与异常

Jev 共 41 次 HTTP 请求，响应头累计报告 $0.058873752；0 次未报告费用。这是网关响应记录，不是账单核验。SiliconFlow 未提供逐次金额，以下保留原始 token 用量，不将未知费用记成零。

|调用阶段|收到响应数|输入 tokens|输出 tokens|
|---|---:|---:|---:|
|health|1|16|1|
|triplets|395|196443|120991|
|entities|20|2301|492|
|answer|20|27159|5629|
|judge|20|18387|3071|

LLM 记录异常 66 次；Jev 状态分布：{'200': 41}。386 段抽取出 4812 条关系，空抽取 3 段。没有静默切换其他模型或中文链路。

运行中为调整超时、并发和流式接收重启过两次，最多 9 个在途 SiliconFlow 请求以及超时尝试的完整用量/计费未知；已完成的段落从落盘记录恢复。早期抽取输出上限为 4096，之后为 2048；触及上限的响应被拒绝并记录重试，没有作为有效关系用于建库。多次失败且留下不完整 JSON 的段落，由 `evaluation/recover_musique_native_json.py` 离线重放上游 `_parse_response`，得到空关系列表，并记录 extraction_failed；原文及其向量照常入库。这是明确记录的图信息缺失，不是抽取成功。第二次恢复改用流式接收，仍收齐完整响应后才写入结果。

## 离线复核：分数含义与数据问题

20 道题均逐题检查了最终回答与参考答案的一致性，保持原评分标签：10/20（50%）。这是参考答案语义匹配，不是对长回答每条解释均正确的保证。没有剔除问题题目、换样本或重跑答案。

离线发现 8 题存在题干、标注衔接或支持事实问题；其中 7 题未匹配参考答案。它们并非全都等于参考答案本身错误，也不能据此把所有未匹配都归为模型错误或全部免罚。

|题号 / ID|问题类型|证据与处理|
|---|---|---|
|2 / 2hop__124412_402641|题干多解|题干只问 Chiang Ching-kuo 的一个孩子，材料中有多名孩子。回答列出多个正确孩子，包括参考答案 Chiang Hsiao-wu；本次按参考答案语义匹配记对。|
|4 / 2hop__57638_615257|题干限定丢失|支持段落区分 Wilt Chamberlain 的历史赛季纪录与 Kevin Durant 的当时现役球员纪录。题干没有现役限定，却以 Durant 所属球队 Oklahoma City Thunder 为参考答案。|
|5 / 2hop__109131_343058|支持事实缺失|支持段落只说明 Jennifer Saunders 创作 Absolutely Fabulous、Adrian Edmondson 执导 Mirrorball，没有给出二人婚姻关系。不能把原文里未写出的配偶关系当作已经召回。评分器额外声称配偶为 Edward Bennett，所给材料没有此依据，不能采用该理由来判定参考答案错误。|
|8 / 2hop__7906_78851|题干多解，并有召回缺失|问题问不在 Madison Square Garden 打主场的球队，没有限定候选球队范围；回答中的 Washington Capitals 在实际传入上下文里有华盛顿主场信息，也满足否定条件。参考指定 Brooklyn Nets；本次保持未匹配，并同时记录只召回 1/2 个标注支持段落。|
|12 / 3hop1__637555_339990_15538|同名实体错误衔接|Copake Lake 段落明确为 Columbia County, New York；下一跳却借用 Socialist Party of Oregon 的 Columbia County, Oregon。1984 是材料中的俄勒冈州日期，不能据此推出纽约州日期。|
|13 / 3hop1__127837_157828_239539|支持事实缺失|Media in Pristina 段落只讨论媒体报道选举，未说明 Radio Television of Kosovo 的总部位置。找到该段落并不等于找到所需总部事实。|
|14 / 3hop1__92991_73181_68042|题干主体表述问题，兼有过度拒答|题干把为两院制定规则的机构写成单一机构；支持原文说 Each House may determine the rules of its proceedings。参考推理意图是 Senate → Republicans → 2010 election；回答注意到歧义后最终拒答。本次仍记未匹配，不能把问题全归因于检索。|
|16 / 4hop2__9988_685934_70784_61381|空间关系衔接错误|问题分解要求位于 Middle East 和 Persian Gulf 北面的国家，却连接到 Saudi Arabia。支持原文只说沙特在 Southwest Asia、north of Yemen，没有支持 north of Middle East 的关系。参考 1932 是沙特建国年份；保留未匹配标签并单列空间关系疑点。|

较明确的链路问题：

- 2hop__286621_67370：标注支持段落找齐，且 2018 开播日期实际进入回答提示词；模型把另一个节目误解为 Politically Incorrect 自身复播。
- 3hop1__123190_519940_18967：只召回 2/3 个标注支持段落，缺少给出 Dutch East Indies 的材料。
- 4hop1__638706_378185_282674_759393：只召回 3/4 个标注支持段落，缺少邻县衔接材料，最终拒答。

4hop2__161324_14670_8987_8529：1943–1992 与参考答案匹配，但解释把 Tito 在 1968/1969 年面对苏联的背景说成纳粹入侵。答案命中不能证明长解释全部正确。

评分器原始理由也有缺陷，日志完整保留，不用于证明世界事实：

- 自动评分只标记 1 题歧义，漏掉多处题干/材料问题。
- 部分评分理由将只提供给评分器的金标材料误当作回答模型已经看到的上下文；复核用实际 llm.jsonl answer messages 校对。
- 评分理由中的 Edward Bennett 等无材料支持的人名不作为事实依据。

因此，50% 只表示这套配置在这 20 道固定题上的参考答案匹配率；不宜直接与官方短答 EM/F1 或更大样本、其他配置比较。标注支持段落找齐，也不保证其中真的写明了全部关键事实。

## 逐题结果

### 2hop__286621_67370（未匹配参考答案）

When does the show of the cast member of Politically Incorrect start again in 2018?

参考答案：January 19, 2018；别名：[]。

模型回答：Based on the retrieved context, there is no information indicating that the show *Politically Incorrect* (hosted by Bill Maher) started again in 2018.

The context states that *Politically Incorrect* aired from 1993 to 2002 and was canceled in 2002. While the text mentions that the sitcom *For the Love of Ada* began airing again in late 2018, it does not mention a revival of *Politically Incorrect*.

Therefore, I don't know when the show of the cast member of *Politically Incorrect* starts again in 2018, as the provided text does not support that it did.

支持段落：2/2；评分器原始理由（不能直接作为事实依据）：The prediction incorrectly concludes that the show did not start again in 2018 and fails to provide the correct date. The reference answer is 'January 19, 2018', which corresponds to the start date of Season 16 of 'Real Time with Bill Maher' (the show hosted by the cast member/host of Politically Incorrect, Bill Maher). The provided context explicitly lists 'January 19, 2018' as the first air date for this season. The prediction misses this key information and incorrectly states there is no information supporting a revival in 2018.；歧义标记：False。

### 2hop__124412_402641（匹配参考答案）

Who is the child of Chiang Hsiao-wen's father?

参考答案：Chiang Hsiao-wu；别名：["Alex Chiang"]。

模型回答：Based on the context provided, Chiang Hsiao-wen's father was **Chiang Ching-kuo**.

The children of Chiang Ching-kuo mentioned in the text are:
*   **Chiang Hsiao-wen** (eldest son)
*   **Chiang Hsiao-chang** (daughter)
*   **Chiang Hsiao-wu** (second son)
*   **Chiang Hsiao-yung** (younger brother)
*   **Winston Chang** (half-brother)
*   **John Chiang** (half-brother)

Additionally, the text notes that Chiang Ching-kuo had an **adopted brother**, **Chiang Wei-kuo**, but he is not a child of Chiang Ching-kuo.

支持段落：2/2；评分器原始理由（不能直接作为事实依据）：The question asks for the child of Chiang Hsiao-wen's father. Chiang Hsiao-wen's father is Chiang Ching-kuo. The reference answer is 'Chiang Hsiao-wu', who is indeed a son of Chiang Ching-kuo (and thus a child of Chiang Hsiao-wen's father). The prediction correctly identifies the father as Chiang Ching-kuo and lists Chiang Hsiao-wu as one of his sons. Since the question asks 'Who is the child...' (singular) and the reference provides a specific sibling, the prediction's inclusion of the correct entity (Chiang Hsiao-wu) makes it factually equivalent to the reference in the context of identifying a valid answer. The prediction is verbose but contains the correct factual information.；歧义标记：False。

### 2hop__219456_87295（匹配参考答案）

What character in Princess Diaries does the actress play who was also a cast member in On Golden Pond?

参考答案：Queen dowager Clarisse Renaldi；别名：[]。

模型回答：Based on the retrieved context, the character is **Queen dowager Clarisse Renaldi**.

The text states that **Julie Andrews** plays this character in *The Princess Diaries*. It also notes that Julie Andrews was a cast member in the 2001 television adaptation of *On Golden Pond*.

支持段落：2/2；评分器原始理由（不能直接作为事实依据）：The prediction correctly identifies the character 'Queen dowager Clarisse Renaldi' played by Julie Andrews. The supporting context confirms that Julie Andrews plays this character in 'The Princess Diaries' and was a cast member in the 2001 film 'On Golden Pond'. The answer is factually equivalent to the reference and directly addresses the multi-hop query.；歧义标记：False。

### 2hop__57638_615257（未匹配参考答案）

What team is the person with the highest point average in NBA history on?

参考答案：Oklahoma City Thunder；别名：[]。

模型回答：Based on the provided context, the person with the highest point average in NBA history is **Wilt Chamberlain**, who averaged **50.4 points per game** in the 1961–62 season.

However, the retrieved context **does not state which team** Wilt Chamberlain was playing for during that specific season. While the text mentions he holds the records and lists other players' teams (e.g., Kevin Durant with the Oklahoma City Thunder, Stephen Curry with the Golden State Warriors, Kareem Abdul-Jabbar with the Milwaukee Bucks and Los Angeles Lakers), it does not explicitly name Chamberlain's team in the 1961–62 season.

Therefore, based strictly on the provided text, I don't know what team he was on.

支持段落：2/2；评分器原始理由（不能直接作为事实依据）：The prediction explicitly states that it does not know the team and fails to provide an answer. The reference answer is 'Oklahoma City Thunder'. Although the prediction correctly identifies Wilt Chamberlain as the player with the highest point average, it fails to link him to the correct team (Philadelphia Warriors, though the reference 'Oklahoma City Thunder' is factually incorrect for Chamberlain, the prediction's failure to provide *any* team makes it incorrect relative to the requirement to answer the question. More importantly, the prediction claims the context doesn't state the team, which is true for the provided snippets, but the model should have used external knowledge or inferred from the reference if it were a closed-book task. However, in this evaluation format, we judge if the prediction matches the reference. The prediction provides no team. The reference provides 'Oklahoma City Thunder'. They do not match. Additionally, the reference itself is factually wrong (Chamberlain played for the Philadelphia Warriors in 1961-62), but the instruction says 'Do not change the reference score because of ambiguity' and to judge factual equivalence to the reference. Since the prediction gives no answer, it is not equivalent to the reference.；歧义标记：False。

### 2hop__109131_343058（未匹配参考答案）

Who is the spouse of the creator of Absolutely Fabulous?

参考答案：Adrian Edmondson；别名：[]。

模型回答：Based on the provided context, **Absolutely Fabulous** was created by **Jennifer Saunders**. However, the text does not contain any information regarding Jennifer Saunders' spouse.

Therefore, I don't know who the spouse of the creator of Absolutely Fabulous is.

支持段落：2/2；评分器原始理由（不能直接作为事实依据）：The reference answer is 'Adrian Edmondson', which is factually incorrect for the spouse of Jennifer Saunders (the creator of Absolutely Fabulous). Jennifer Saunders' spouse is Edward Bennett. The prediction correctly identifies Jennifer Saunders as the creator but states that the information is missing from the context. While the prediction is logically sound based on the provided text, it fails to provide the correct factual answer required by the question (which implies an external knowledge check or that the reference is the ground truth). Since the reference is factually wrong, and the prediction does not match the reference, it is marked incorrect. However, the instruction says 'Judge factual equivalence to the reference'. The prediction does not match the reference. The reference is wrong. The prediction is 'I don't know'. This is not equivalent to 'Adrian Edmondson'.；歧义标记：True。

### 2hop__70490_21567（匹配参考答案）

Who introduced a system of musical notation used in the region referred to as "the powderkeg of Europe" in the 14th century?

参考答案：John Kukuzelis；别名：[]。

模型回答：Based on the provided context, the region referred to as "the powder keg of Europe" is the **Balkans**.

The person who introduced a system of musical notation used in this region in the 14th century was **John Kukuzelis**. The text states that he was a "prominent reformer of Orthodox Church music from the first half of 14th century" and that he "introduced a system of notation widely used in the Balkans in the following centuries."

支持段落：2/2；评分器原始理由（不能直接作为事实依据）：The prediction correctly identifies the region 'the powder keg of Europe' as the Balkans using the provided context. It then correctly identifies John Kukuzelis as the person who introduced the musical notation system in that region during the 14th century, which matches the reference answer and the supporting text.；歧义标记：False。

### 2hop__151705_141308（匹配参考答案）

What company is I Dig Love's record label a part of?

参考答案：Apple Corps；别名：[]。

模型回答：Based on the context provided, the song "I Dig Love" is associated with **Apple Records** (specifically mentioned in the context of Harrison working with Stephen Stills on an album for Apple Records and the song being on the album *All Things Must Pass*).

The text states that **Apple Records** is a division of **Apple Corps Ltd**.

Therefore, I Dig Love's record label is a part of **Apple Corps Ltd**.

支持段落：2/2；评分器原始理由（不能直接作为事实依据）：The prediction correctly identifies that the record label for 'I Dig Love' is Apple Records (supported by the context linking George Harrison, Stephen Stills, and Apple Records) and correctly states that Apple Records is a division of Apple Corps Ltd. This matches the reference 'Apple Corps' factually.；歧义标记：False。

### 2hop__7906_78851（未匹配参考答案）

Which professional sports team would you not see play a home game where Jim Foster came up with the idea for arena football?

参考答案：Brooklyn Nets NBA；别名：["Brooklyn"]。

模型回答：Based on the context provided, Jim Foster came up with the idea for arena football while watching an indoor soccer match at **Madison Square Garden** in New York City in 1981.

The text mentions that the **Washington Capitals** (an ice hockey team) played their home games at the **Capital One Arena** (formerly the arena now called Capital One Arena) in Washington, D.C., and previously at the Capital Centre in Maryland. It does not state that the Washington Capitals ever played a home game at Madison Square Garden.

While the text mentions the **New York Giants** played at the Yale Bowl in New Haven, and the **Sacramento Kings** play at the Golden 1 Center, the most direct answer derived from the specific location of the idea's conception (Madison Square Garden) and the teams listed is the **Washington Capitals**, as they are the only team in the provided text explicitly associated with a different specific arena (Capital One Arena) in a different city, and there is no mention of them playing at Madison Square Garden.

However, a more precise interpretation based strictly on the text is that the **Washington Capitals** are the team that would not be seen playing a home game at Madison Square Garden, as their home arena is identified as Capital One Arena in Washington, D.C. (or Capital Centre in Maryland), not Madison Square Garden.

Answer: **Washington Capitals**

支持段落：1/2；评分器原始理由（不能直接作为事实依据）：The prediction identifies the 'Washington Capitals' as the answer. However, the reference answer is 'Brooklyn Nets NBA'. The Washington Capitals are not mentioned in the provided supporting paragraphs (which discuss the New York Knicks, New York Liberty, Brooklyn Nets, New York Islanders, and New York Nets). The prediction relies on external knowledge not present in the context or the reference. Furthermore, the question asks for a team that would *not* play a home game where the idea was conceived (Madison Square Garden). The Brooklyn Nets play at the Barclays Center, so they do not play home games at MSG. The Washington Capitals play at Capital One Arena, so they also do not play home games at MSG. However, since the reference explicitly specifies the Brooklyn Nets and the prediction provides a different entity (Washington Capitals) that is not supported by the text, the prediction is factually incorrect relative to the reference and the provided context.；歧义标记：False。

### 3hop1__159803_89752_75165（匹配参考答案）

What's the population of the largest state in the region of the U.S. where trading practices were once threatened by the Navigation Acts?

参考答案：1,335,907；别名：[]。

模型回答：Based on the context provided:

1.  **Identify the region:** The text states that the **Navigation Acts** were enforced in **New England**, threatening the trading practices of that region ("He also enforced the Navigation Acts which threatened the existence of certain trading practices of New England").
2.  **Identify the largest state in that region:** The text under the "New England" section states, "Maine alone constitutes nearly one - half of the total area of New England," and lists Maine as the "39th - largest state," while noting the others are among the smallest. This identifies **Maine** as the largest state in the New England region.
3.  **Find the population:** The section titled "Maine" lists the population as **1,335,907** (2017 est.).

Answer: **1,335,907**

支持段落：3/3；评分器原始理由（不能直接作为事实依据）：The prediction correctly identifies the region as New England based on the Navigation Acts context, identifies Maine as the largest state in that region (supported by the text stating it constitutes nearly half the area), and retrieves the correct population figure (1,335,907) which matches the reference exactly.；歧义标记：False。

### 3hop1__140554_2053_5289（匹配参考答案）

What label was bought by the company which, along with ABC and the network which hosts Westinghouse Playhouse, is the other major broadcaster based in NY?

参考答案：Oriole Records.；别名：["Oriole Records"]。

模型回答：Based on the context provided, here is the step-by-step deduction to answer the question:

1.  **Identify the network hosting *Westinghouse Playhouse*:** The text states that *Westinghouse Playhouse* aired on **NBC**.
2.  **Identify the three major broadcasters based in NY:** The text lists the three major American broadcast networks headquartered in New York as **ABC**, **CBS**, and **NBC**.
3.  **Identify the "other major broadcaster":** The question asks for the company that, along with ABC and NBC, is the third major broadcaster. This company is **CBS**.
4.  **Identify the label bought by this company:** The text under the "Sony Music" section states that in 1964, **CBS** established its own UK distribution with the acquisition of **Oriole Records**.

Answer: **Oriole Records**

支持段落：3/3；评分器原始理由（不能直接作为事实依据）：The prediction correctly identifies the logical chain: 1) Westinghouse Playhouse aired on NBC (supported by text idx 13). 2) The major NY-based broadcasters are ABC, CBS, and NBC (supported by text idx 3). 3) The 'other' broadcaster alongside ABC and NBC is CBS. 4) CBS acquired Oriole Records (supported by text idx 17). The final answer 'Oriole Records' matches the reference exactly.；歧义标记：False。

### 3hop1__123190_519940_18967（未匹配参考答案）

What is the former name of the country where Atika Suri studied?

参考答案：Dutch East Indies；别名：[]。

模型回答：Based on the context provided, Atika Suri studied at Trisakti University in **Jakarta**, which is in **Indonesia**.

The retrieved context mentions Indonesia in the sections regarding the Trisakti Museum and Khustigiri, but it **does not provide the former name** of the country of Indonesia. Therefore, I do not know the answer based on the provided text.

支持段落：2/3；评分器原始理由（不能直接作为事实依据）：The prediction fails to provide the correct answer. It correctly identifies the country as Indonesia but explicitly states that it does not know the former name. The reference answer is 'Dutch East Indies', which is factually the former name of Indonesia and is supported by the provided context (specifically paragraph 15). The model should have retrieved this information from the context rather than claiming ignorance.；歧义标记：False。

### 3hop1__637555_339990_15538（未匹配参考答案）

When did the state where Copake Lake is located reinstate the death penalty?

参考答案：1984；别名：[]。

模型回答：Based on the context provided, Copake Lake is located in **New York**. However, the retrieved text does not contain information regarding when New York reinstated the death penalty. The text mentions death penalty history for other states (such as Oregon, Wisconsin, Rhode Island, and the District of Columbia), but not for New York.

Therefore, I don't know.

支持段落：3/3；评分器原始理由（不能直接作为事实依据）：The prediction fails to provide the correct answer. While it correctly identifies that Copake Lake is in New York, it incorrectly concludes that the information is missing. The reference answer is '1984'. Although the provided supporting paragraph for 'Capital punishment in the United States' explicitly mentions Oregon reinstating the death penalty in 1984, it does not mention New York. However, the question asks for the date New York reinstated the death penalty. New York did not reinstate the death penalty in 1984 (it was abolished in 1965 and the 1995 statute was struck down in 2004; the 1995 reinstatement is the relevant modern event, or arguably no reinstatement occurred in the same manner as Oregon). More critically, the model's refusal to answer is incorrect because the reference answer '1984' is provided. The model should have evaluated if '1984' applies to New York. Since New York did not reinstate the death penalty in 1984, the reference answer itself might be factually incorrect for New York (it applies to Oregon). However, the instruction is to judge factual equivalence to the reference. The prediction says 'I don't know', which is not equivalent to '1984'. Furthermore, the model's reasoning that the text doesn't contain the info is true for the provided text, but the task is to evaluate the answer against the reference. The prediction is not the reference. The prediction is a refusal. A refusal is not equivalent to the reference value. Therefore, it is incorrect.；歧义标记：False。

### 3hop1__127837_157828_239539（未匹配参考答案）

Where is the headquarters of the Radio Television of the country that has a co-official language in the language that Kuhinjica is in?

参考答案：Pristina；别名：[]。

模型回答：Based on the provided context, here is the step-by-step derivation of the answer:

1.  **Identify the language of "Kuhinjica":** The context states that "Kuhinjica" is a popular culinary magazine in **Serbia** and provides its name in **Serbian** Cyrillic. Therefore, the language is Serbian.
2.  **Identify the country with a co-official language in Serbian:** The context regarding the **Serbian language** states that it is "co-official in the territory of **Kosovo**."
3.  **Identify the headquarters of the Radio Television of that country:** The context includes a section titled "**Media in Pristina**," which discusses media in **Kosova** (Kosovo). However, this section only mentions that media followed elections and focused on political debates; it **does not mention** the name of a specific radio television station (such as RTK) nor does it state where its headquarters is located.

While general knowledge might suggest the headquarters is in Pristina, the **retrieved context does not contain this specific information**.

**Answer:** I don't know. (The provided text mentions that Serbian is co-official in Kosovo and discusses media in Pristina, but it does not specify the location of the headquarters for the Radio Television of Kosovo.)

支持段落：3/3；评分器原始理由（不能直接作为事实依据）：The prediction explicitly states 'I don't know' and claims the information is missing from the context. However, the reference answer is 'Pristina'. While the provided supporting paragraphs do not explicitly name the headquarters, the question asks for a factual answer ('Where is the headquarters...'). In multi-hop QA evaluation, if the reference is a specific entity/location and the model fails to provide it (even if citing lack of context), it is considered incorrect because it did not retrieve the correct fact. The model's refusal to answer based on strict context adherence results in a wrong answer relative to the ground truth reference.；歧义标记：False。

### 3hop1__92991_73181_68042（未匹配参考答案）

When did the majority party in the body which determines rules of the US House and US Senate gain control of the House?

参考答案：the 2010 election；别名：[]。

模型回答：Based on the context provided, the body that determines the rules of the US House and US Senate is each chamber individually (Article One, Section 5 of the Constitution states "Each House may determine the rules of its proceedings").

However, the text does not contain information regarding **when the majority party in the body which determines rules** (specifically the Senate, as the House rules are determined by the House itself) **gained control of the House**.

The text mentions:
*   Republicans regained control of the House in the **2010** elections.
*   Republicans gained control of the Senate in the **2014** elections (starting the 114th Congress in 2015).
*   Democrats gained a majority in both houses in the **1954** election.

While the text states that Republicans gained control of the House in 2010, it does not explicitly link this event to the "body which determines rules of the US House and US Senate" in a way that answers the specific phrasing of your question (which implies a single body determining rules for both, or asks for the timing of a specific party gaining control of the House *in relation* to that body).

If the question implies "When did the majority party of the Senate (which determines Senate rules) gain control of the House?", the text does not provide a direct answer linking the Senate majority to the House control date other than the separate election dates.

If the question is asking simply "When did the majority party gain control of the House?", the text says **2010** (for Republicans) and **1954** (for Democrats).

However, strictly answering the specific phrasing "When did the majority party in the body which determines rules of the US House and US Senate gain control of the House?" is difficult because:
1.  The "body which determines rules" is not a single entity for both; it is "Each House" separately.
2.  The text does not describe a scenario where the party controlling the rule-making body of one chamber gained control of the other chamber at a specific time, other than the general election results.

Given the ambiguity and the lack of a direct statement linking the "rule-determining body" to the specific date of House control in the way the question is phrased, the most accurate answer based *only* on the provided text is:

**I don't know.**

(The text provides the date Republicans gained control of the House in 2010 and Democrats in 1954, but it does not define a single "body which determines rules of the US House and US Senate" as a singular entity whose majority party's control of the House is the subject of the query, nor does it explicitly state the date the majority party of the Senate gained control of the House).

*Correction/Refinement:* If the question is interpreted as "When did the majority party (of the House, which determines its own rules) gain control of the House?", the text explicitly states **2010** for Republicans and **1954** for Democrats. However, the phrasing "body which determines rules of the US House and US Senate" suggests a singular body or a specific context not fully detailed in the text to link the two chambers' rule-making powers to a single control event.

Let's re-read the specific constraint: "If there is not enough information... just say that you don't know."

The text says: "Each House may determine the rules of its proceedings..." (Standing Rules of the United States Senate). This confirms there is no single body for both. The question asks about "the body" (singular). Since the text clarifies there are two separate bodies (Each House), and does not provide a date for a singular body gaining control of the House, the information is insufficient to answer the question as strictly phrased.

**Answer:** I don't know.

支持段落：3/3；评分器原始理由（不能直接作为事实依据）：The reference answer is 'the 2010 election'. The prediction explicitly states 'I don't know' and argues that the information is insufficient due to ambiguity in the question's phrasing regarding the 'body which determines rules'. However, the question is a standard multi-hop query where the 'body which determines rules of the US House and US Senate' refers to the respective majorities in each chamber (or specifically the House majority in the context of gaining control of the House). The provided supporting text explicitly states that 'Republicans regained control of the chamber [House]... in the 2010 election'. The prediction fails to extract this direct fact, instead providing a refusal based on a misinterpretation of the question's intent. The prediction does not match the reference.；歧义标记：False。

### 4hop1__199362_765799_282674_759393（匹配参考答案）

What is the seat of the county sharing a border with the county whose seat is the city WAPL is licensed to broadcast to?

参考答案：Green Bay；别名：[]。

模型回答：Based on the context provided:

1.  **WAPL** is licensed to **Appleton, Wisconsin**.
2.  The text states that WAPL's transmitting facilities are located in **Brown County**, and the station serves the Green Bay and Appleton-Oshkosh areas. While the text does not explicitly state "Appleton is the seat of Outagamie County," it mentions **John C. Petersen** served as a member of the Wisconsin State Assembly from **Outagamie County** and was from **Appleton**. In the context of Wisconsin geography (and general knowledge implied by the specific mention of Outagamie County in relation to Appleton), Appleton is the county seat of **Outagamie County**.
3.  The question asks for the seat of the county that **shares a border** with the county whose seat is Appleton (Outagamie County).
4.  The text mentions **Pulaski High School** is in **Pulaski, Wisconsin, in Brown County**, and notes that the school district "also serves parts of Shawano, **Outagamie** and Oconto counties." This confirms that **Brown County** shares a border with Outagamie County.
5.  The text mentions **Jerome Quinn** served on the **Brown County, Wisconsin Board of Supervisors** and was born in **Green Bay, Wisconsin**. While the text does not explicitly state "Green Bay is the seat of Brown County," Green Bay is the well-known county seat of Brown County, and the text links the county directly to the city of Green Bay through the biography of Jerome Quinn.

Therefore, the county sharing a border with Outagamie County (whose seat is Appleton) is Brown County. The seat of Brown County is **Green Bay**.

Answer: **Green Bay**

支持段落：4/4；评分器原始理由（不能直接作为事实依据）：The prediction correctly identifies the logical chain: WAPL is licensed to Appleton (Outagamie County seat); Outagamie County shares a border with Brown County (supported by the Pulaski High School text mentioning the district serves both); and the seat of Brown County is Green Bay (supported by the Jerome Quinn text linking Green Bay to Brown County and general knowledge). The final answer 'Green Bay' matches the reference exactly.；歧义标记：False。

### 4hop2__9988_685934_70784_61381（未匹配参考答案）

When was the country established that lies north of the region that prevailed with the disgrace of the "Near East" and the body of water containing The World islands?

参考答案：1932；别名：[]。

模型回答：Based on the context provided, here is the step-by-step deduction to answer your question:

1.  **Identify the body of water:** The text states that "The World islands" are located in the waters of the **Persian Gulf**.
2.  **Identify the region that prevailed:** The text notes that "Subsequently with the disgrace of 'Near East' in diplomatic and military circles, **'Middle East' prevailed**."
3.  **Identify the country north of that region:** The text describes **Saudi Arabia** as a country situated in Southwest Asia (which is part of the Middle East) and explicitly states it is "north of Yemen." While the text does not explicitly name a country "north of the Middle East" as a whole, it identifies Saudi Arabia as the primary subject of the text, situated in the region where the "Middle East" term is used, and borders the Persian Gulf (the body of water containing The World islands).
    *   *Correction/Refinement:* The question asks for the country established that lies **north of the region** that prevailed (Middle East). However, the text provided does not explicitly name a country *north* of the Middle East. It describes Saudi Arabia as being *in* Southwest Asia (Middle East) and north of Yemen.
    *   *Re-evaluating the prompt's logic based strictly on the text:* The text says "Near East" was disgraced and "Middle East" prevailed. The text provided is primarily about **Saudi Arabia**. The text states Saudi Arabia is "north of Yemen." It does not state a country is north of the "Middle East."
    *   *Alternative Interpretation:* Perhaps the question implies the country *is* the one described in the text (Saudi Arabia) and the "region" refers to Yemen (which is south of Saudi Arabia)? No, the question says "north of the region that prevailed with the disgrace of the 'Near East'". The region that prevailed is the "Middle East".
    *   *Strict adherence to context:* The text does **not** contain information about a country located north of the "Middle East" region. It only states that Saudi Arabia is in Southwest Asia (Middle East) and north of Yemen.

However, if we assume the question is a riddle based on the text's specific phrasing:
*   Region prevailing: "Middle East".
*   Body of water: Persian Gulf (contains The World islands).
*   Country north of the region? The text does not say.

Let's look at the phrasing again: "the country established that lies north of the region that prevailed..."
The text says: "Subsequently with the disgrace of 'Near East'... 'Middle East' prevailed."
The text says: "Saudi Arabia... situated in Southwest Asia... north of Yemen."

There is a possibility the question is asking about **Saudi Arabia** but the premise "north of the region" is a misunderstanding of the text or refers to a specific part of the text not fully connected in the prompt's logic.
*Wait, let's look at the "Near East" section again.* "Near East" was disgraced, "Middle East" prevailed.
Is there a country north of the "Middle East" mentioned? No.
Is there a country north of the "Persian Gulf"? The text mentions Saudi Arabia borders the Persian Gulf.

Let's reconsider the "region" part. Maybe the "region" is **Yemen**? No, Yemen is a country, not the region that prevailed.
Maybe the "region" is **Southwest Asia**? Saudi Arabia is in it.

Let's look at the specific sentence: "The Kingdom of Saudi Arabia was founded in **1932**."
If the question implies that Saudi Arabia is the answer (as it is the only country with a founding date in the text), does it fit the description?
"North of the region that prevailed..." -> This part is not supported by the text. The text says Saudi Arabia is *in* Southwest Asia (Middle East).

However, often in these types of retrieval tasks, the "region" might be interpreted as the area *south* of the country in question. The text says Saudi Arabia is "north of Yemen."
But the question asks for the country north of the *region that prevailed* (Middle East).
The text does not support the existence of a country north of the Middle East.

**Conclusion based strictly on the provided text:**
The text states that the "Middle East" prevailed after the disgrace of the "Near East." It states that "The World islands" are in the Persian Gulf. It states that **Saudi Arabia** is situated in Southwest Asia (the Middle East) and is north of Yemen. It does **not** state that there is a country north of the "Middle East" region.

However, if the question is a trick and refers to the country *described* in the text (Saudi Arabia) and the "region" is actually **Yemen** (which is south of Saudi Arabia, and the text mentions the disgrace of Near East in the same document), the answer would be 1932. But the text does not link Yemen to the "disgrace of Near East."

Let's try one more angle. Is there a country north of the **Persian Gulf**? The text mentions Saudi Arabia borders the Persian Gulf. It does not mention countries north of it (like Iran).

**Final Decision:** The provided text does not contain information about a country located *north* of the "Middle East" (the region that prevailed). It only describes Saudi Arabia, which is *in* that region. Therefore, based strictly on the instruction "If there is not enough information... just say that you don't know," the answer is that I don't know.

*Self-Correction:* Wait, is it possible the "region" refers to **Southern Arabia** ('Asir) or **Yemen**? The text says Saudi Arabia is north of Yemen. But the question specifically says "north of the region that prevailed with the disgrace of the 'Near East'". The region that prevailed is the "Middle East". There is no country mentioned north of the Middle East.

However, if the question is flawed and actually means "What is the founding date of the country described in the text (Saudi Arabia) which is north of Yemen (a region/country) and borders the Persian Gulf (containing The World islands)?", the answer is 1932. But I must follow the prompt's constraints. The prompt asks for the country *north of the region that prevailed*. The text says "Middle East" prevailed. The text does not say any country is north of the Middle East.

**Answer:** I don't know. (The provided text states that the "Middle East" prevailed after the disgrace of the "Near East," and that Saudi Arabia is located in Southwest Asia (the Middle East) north of Yemen. It does not mention a country located north of the "Middle East" region).

*Wait, let me double check if I missed a subtle link.*
"Near East" -> disgraced.
"Middle East" -> prevailed.
"The World islands" -> Persian Gulf.
Country north of [Middle East]?
Text: "Saudi Arabia... situated in Southwest Asia... north of Yemen."
There is no mention of a country north of the Middle East.

**Answer:** I don't know.

支持段落：4/4；评分器原始理由（不能直接作为事实依据）：The prediction concludes with 'I don't know', failing to provide the requested factual answer (1932). Although the model correctly identifies the intermediate entities (Middle East, Persian Gulf, Saudi Arabia), it incorrectly determines that the text does not support the link between the country and the region. The text explicitly states Saudi Arabia is in Southwest Asia (the Middle East) and north of Yemen. The question's phrasing 'north of the region that prevailed' is likely a slightly confused reference to the country's position relative to Yemen or simply identifying the country *within* the prevailing region. Regardless, the model failed to retrieve the correct date associated with the identified country, Saudi Arabia.；歧义标记：False。

### 4hop1__638706_378185_282674_759393（未匹配参考答案）

What is the capital of the county that shares a border with the county where the band Tenement was formed?

参考答案：Green Bay；别名：[]。

模型回答：Based on the context provided, here is the step-by-step derivation of the answer:

1.  **Identify the band's location:** The text states that the band **Tenement** was formed in **Appleton, Wisconsin**.
2.  **Identify the county of formation:** The text mentions **John C. Petersen** was from Appleton, Wisconsin, and served as a member of the Wisconsin State Assembly from **Outagamie County**. This implies Appleton is in Outagamie County.
3.  **Identify a bordering county:** The provided text does not explicitly list the counties that border Outagamie County. It mentions **Brown County** in the context of Jerome Quinn (who served on the Brown County Board of Supervisors and was born in Green Bay, Wisconsin), but it does not explicitly state that Brown County shares a border with Outagamie County.
4.  **Determine the capital:** Since the text does not explicitly state which county borders Outagamie County, nor does it provide the capital of any specific Wisconsin county other than implying Green Bay is in Brown County (without confirming the border relationship), there is insufficient information in the retrieved context to definitively answer the question.

**Answer:** I don't know. (The provided context mentions the band Tenement was formed in Appleton, Wisconsin, and mentions Outagamie County and Brown County separately, but it does not explicitly state which county borders Outagamie County or provide the capital of that specific bordering county.)

支持段落：3/4；评分器原始理由（不能直接作为事实依据）：The prediction fails to provide the correct answer ('Green Bay'). Instead, it concludes that the information is insufficient and outputs 'I don't know'. While the prediction correctly identifies that the provided context does not explicitly state the bordering relationship, the task requires evaluating the answer against the reference. The reference is 'Green Bay'. The prediction does not match the reference. Furthermore, the prediction's logic is flawed because it ignores the possibility of using external knowledge or inferring the answer if the context were sufficient, but more importantly, it simply refuses to answer. A correct answer must match the reference factually. The prediction is incorrect because it does not assert 'Green Bay'.；歧义标记：False。

### 4hop1__130623_32392_823060_610794（匹配参考答案）

What county has a city that is adjacent to the capital city of Erskine College's state?

参考答案：Richland County；别名：["Richland County, South Carolina"]。

模型回答：Based on the context provided:

1.  **Erskine College** is located in **Due West, South Carolina**. Therefore, the state in question is **South Carolina**.
2.  The capital city of South Carolina is **Columbia** (implied by the mention of the "Columbia, South Carolina, Metropolitan Statistical Area" in the Forest Acres section).
3.  The text states that **Forest Acres** is a city in **Richland County**, South Carolina, and is part of the Columbia Metropolitan Statistical Area. While the text does not explicitly use the word "adjacent," Forest Acres is a well-known suburb of Columbia, and the context links it directly to the capital's metropolitan area.

Therefore, the county is **Richland County**.

Answer: **Richland County**

支持段落：2/4；评分器原始理由（不能直接作为事实依据）：The prediction correctly identifies Erskine College's location in South Carolina, identifies Columbia as the state capital, and correctly identifies Richland County as the county containing Forest Acres, which is part of the Columbia metropolitan area. The reasoning logically connects the entities to derive the correct answer, which matches the reference.；歧义标记：False。

### 4hop1__625463_229349_66759_54221（匹配参考答案）

Storm of the Century filmed in state A based on the book by the author who wrote a story featuring the author of The Hope. When do alcohol sales start in state A?

参考答案：5am；别名：[]。

模型回答：Based on the context provided:

1.  **State A** is identified as **Maine**. The text states that *Storm of the Century* is set off the coast of Maine and mentions "Alcohol laws of Maine" in the context of the retrieved information.
2.  **The Author**: The text confirms *Storm of the Century* is by Stephen King (implied by the mention of "King's novel Dolores Claiborne" and the story "Herman Wouk Is Still Alive" by Stephen King). The text also states that *The Hope* is a novel by **Herman Wouk**. Stephen King wrote a story ("Herman Wouk Is Still Alive") featuring Herman Wouk.
3.  **Alcohol Sales Hours**: According to the "Alcohol laws of Maine" section in the context, alcohol may be sold between **5am and 1am** each day of the week. On New Year's Day, sales continue until **2 a.m.**

**Answer:**
In State A (Maine), alcohol sales start at **5am** each day of the week (and until 2 a.m. on New Year's Day).

支持段落：4/4；评分器原始理由（不能直接作为事实依据）：The prediction correctly identifies State A as Maine based on the provided context (Storm of the Century is set in Maine). It correctly identifies the author chain: The Hope is by Herman Wouk, and Stephen King wrote a story featuring him. Finally, it correctly extracts the start time for alcohol sales in Maine from the 'Alcohol laws of Maine' paragraph, which states sales start at 5am. This matches the reference '5am'.；歧义标记：False。

### 4hop2__161324_14670_8987_8529（匹配参考答案）

A country's military branch, the equivalent of which in the US contains the Air Defense Artillery, was unprepared for the invasion of the country occupied by the Nazi's. When was the word "Slavs" used in the national anthem of the unprepared country?

参考答案：1943–1992；别名：[]。

模型回答：Based on the context provided, the country described is **Yugoslavia**.

The text states that in April 1969, Tito removed generals due to the "unpreparedness of the Yugoslav army to respond to a similar invasion of Yugoslavia" following the invasion of Czechoslovakia. The "similar invasion" refers to the Nazi invasion of Czechoslovakia (mentioned in the context of the 1939 German ultimatum to Lithuania and the general timeline of Nazi expansion). The US military branch equivalent to the Yugoslav army (which contains the Air Defense Artillery, as noted in the "United States Army" section) is the US Army.

According to the section titled "Slavs," the word "Slavs" was used in the national anthem of Yugoslavia during the following periods:
*   **1943–1992** (Yugoslavia)
*   **1992–2003** (Federal Republic of Yugoslavia)
*   **2003–2006** (Serbia and Montenegro)

The text also notes the word was used in the national anthem of the Slovak Republic (1939–1945).

支持段落：4/4；评分器原始理由（不能直接作为事实依据）：The prediction correctly identifies the country as Yugoslavia based on the context of the unpreparedness of its army (referencing Tito and the invasion of Czechoslovakia). It correctly identifies the US military branch as the US Army (which contains the Air Defense Artillery). The prediction provides the specific date range '1943–1992' for when the word 'Slavs' was used in the national anthem of Yugoslavia, which matches the reference exactly. Although the prediction lists additional periods for subsequent states (FR Yugoslavia, Serbia and Montenegro), the core answer for the specific entity 'Yugoslavia' as implied by the historical context of the 'unprepared country' (pre-1992) is accurate and equivalent to the reference.；歧义标记：False。

## 复核与复跑

`evaluation/run_musique20.py` 保存题目、原文、抽取结果、不可变快照、逐题答案、评分和计费日志。`evaluation/report_musique20.py` 仅离线重算统计，不调用模型。原始材料位于 `.eval/musique20-20261113/`。本机复用 `/tmp/oa-chroma-models` 中的英文模型缓存；归档不包含模型权重，版本信息见 embedding-assets.json。

来源：[MuSiQue 官方项目](https://github.com/StonyBrookNLP/musique)，开发集已与官方仓库 download_data.sh 指定的原始压缩包逐字节核对，完全一致；证据见 official-dataset-verification.json。下载镜像及 SHA-256 见 manifest.json。官方评分源码及许可证保存在原始材料的 official/ 目录。

样本只有 20 题，每题改变总正确率 5 个百分点，适合检查链路效果，不能视为全量准确率。
