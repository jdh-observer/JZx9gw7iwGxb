---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["title"] -->
# Mapping the Latent Past: Assessing Large Language Models as Digital Tools through Source Criticism
<!-- #endregion -->

<!-- #region editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["contributor"] -->
 ### anonym


<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} tags=["copyright"] -->
[![cc-by](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/) 
©<AUTHOR or ORGANIZATION / FUNDER>. Published by De Gruyter in cooperation with the University of Luxembourg Centre for Contemporary and Digital History. This is an Open Access article distributed under the terms of the [Creative Commons Attribution License CC-BY](https://creativecommons.org/licenses/by/4.0/)

<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} tags=["copyright"] -->
[![cc-by-nc-nd](https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-nd/4.0/) 
©<AUTHOR or ORGANIZATION / FUNDER>. Published by De Gruyter in cooperation with the University of Luxembourg Centre for Contemporary and Digital History. This is an Open Access article distributed under the terms of the [Creative Commons Attribution License CC-BY-NC-ND](https://creativecommons.org/licenses/by-nc-nd/4.0/)

<!-- #endregion -->


<!-- #region editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["keywords"] -->
Large language models, Artifical intelligence, Generative AI, Benchmarking, Optical character recognition, Oral history, Prompt engineering
<!-- #endregion -->

<!-- #region editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["abstract"] -->
This article examines how digital historians can use large language models (LLMs) as research tools while critically assessing their limitations through source criticism of their underlying training data. Case studies of LLM performance on historical knowledge benchmarks, oral history transcriptions, and OCR corrections reveal how these technologies encode patterns of whose history has been digitized and made computationally legible. These variations in performance across linguistic and temporal domains reveal the uneven terrain of knowledge encoded within generative AI systems. By mapping this "jagged frontier" of AI capabilities, historians can evaluate LLMs not just as tools but as historical sources shaped by the scale and diversity of their training. The article concludes by examining how historians can develop new forms of source criticism to navigate generative AI's uneven potential while contributing to broader debates about these technologies' societal impact.
<!-- #endregion -->

<!-- #region editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} -->
## Introduction
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"muld2": [{"id": "20666258/7M6MP3NI", "source": "zotero"}], "tfptr": [{"id": "20666258/37INR4W2", "source": "zotero"}], "tnaq8": []}} editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} -->
In 2003, Roy Rosenzweig predicted that digital historians would need to develop new techniques "to research, write, and teach in a world of unheard-of historical abundance." (<cite id="tfptr"><a href="#zotero%7C20666258%2F37INR4W2">(Rosenzweig, 2003)</a></cite>) Over the past two decades historians have risen to this challenge, embracing digital mapping, network analysis, distant reading of large text collections, and machine learning as part of a growing methodological toolkit. (<cite id="muld2"><a href="#zotero%7C20666258%2F7M6MP3NI">(Graham et al., 2015)</a></cite>) The use of these tools also revealed that every approach possesses distinct strengths and weaknesses, qualities informed not only by practical use but also by critical and ethical perspectives. Generative artificial intelligence (AI) has emerged as another potential tool for historians, particularly large language models (LLMs), the most prominent form of this technology. These models possess striking capacities to generate, interpret, and manipulate data across a range of modalities. The rapidly-expanding scope of these capabilities and their limits remain intensely debated, as do their broader social, economic, cultural, and environmental impacts. Yet while still an emerging technology, historians are already demonstrating generative AI's potential as a versatile digital tool. Historians are also contributing to the critical discourse surrounding this new domain, raising key questions about how these models are created, their propensity to reinforce existing inequalities, and their potential to distort our understanding of the past. (<cite id="tnaq8"><a href="#zotero%7C20666258%2FPCJH9RBZ">(Meadows &#38; Sternfeld, 2023)</a></cite>)

<!-- #endregion -->

<!-- #region citation-manager={"citations": {"51rof": [{"id": "20666258/9IDUQQET", "source": "zotero"}], "5byan": [{"id": "20666258/IXVBRSGM", "source": "zotero"}], "wtaoc": [{"id": "20666258/2GJME5SQ", "source": "zotero"}]}} editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} -->
Sarah Oberbichler and Cindarella Petz advance this dialogue by distinguishing between AI applications as tools that support existing research processes and AI as methods that shape historical interpretation itself. They convincingly argue that "researchers using AI as a supporting tool need basic AI literacy to
effectively utilize these technologies while maintaining critical awareness of their limitations." (<cite id="5byan"><a href="#zotero%7C20666258%2FIXVBRSGM">(Oberbichler &#38; Petz, 2025)</a></cite>) This distinction becomes particularly salient when considering LLMs as a new form of historical source. The contours of these sources can be read through the nature of their training, the data hierarchies encoded within them, and the patterns that inform their responses. LLMs represent an algorithmic cartography of our collective digital culture, featuring prominent peaks of capability as well as uncanny valleys of stochastic distortion. In making these sources legible we can examine how these models are influenced by their time and place, and anchored by a particular and often distorted view of the world. Exploring this uneven terrain represents an approach Ted Underwood has described as mapping the "latent space" of generative AI, where these computational landscapes can be utilized for scholarly ends. (<cite id="51rof"><a href="#zotero%7C20666258%2F9IDUQQET">(Underwood, 2021)</a></cite>) However, as Frédéric Clavert reminds us, this terrain is not a neutral expanse. It is instead a “grid of interpretations,” a matrix by which underlying cultural hierarchies determine how models organize and recombine the basis for their abilities: their training data. As historians explore and critique generative AI, understanding this relationship offers important insights into its potential use. (<cite id="wtaoc"><a href="#zotero%7C20666258%2F2GJME5SQ">(Clavert, 2024)</a></cite>)
<!-- #endregion -->

<!-- #region editable=true jupyter={"outputs_hidden": false} raw_mimetype="" slideshow={"slide_type": ""} -->
## What Do AIs "Know" About History? Evaluating LLMs as Historical Sources
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"1vggn": [{"id": "20666258/RRUNZDAM", "source": "zotero"}], "2f6wb": [], "3yqem": [{"id": "20666258/7ERZCN5G", "source": "zotero"}], "jq1bi": [{"id": "20666258/WHCGSCI5", "source": "zotero"}], "k2mij": [{"id": "20666258/Q269X8CB", "source": "zotero"}], "m0i3p": [{"id": "20666258/MAEXPBX2", "source": "zotero"}], "o5ac6": [{"id": "20666258/P96ZKU8N", "source": "zotero"}], "rhbf3": [{"id": "20666258/8ISS2NP3", "source": "zotero"}]}} editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} -->
Mapping this algorithmic cartography requires an understanding of how these models learn about the past and what they “know” about history; and perhaps more importantly, whose past has been learned and whose history remains obscured. A useful tool in reading these models as sources are technical assessments of their capacities for historical knowledge. By examining where models excel and where they falter, we can trace how their training data amplifies certain voices while silencing others. Such assessments can inform the use of generative AI for historical ends.  

At the most fundamental level generative AI models are statistical representations of the datasets on which they are trained. Machine learning techniques like deep learning and innovations like the Transformer network architecture have enabled the development of models that can extract and encode patterns from massive datasets. (<cite id="o5ac6"><a href="#zotero%7C20666258%2FP96ZKU8N">(Vaswani et al., 2023)</a></cite>) Researchers have further discovered that these models exhibit a range of “emergent” capabilities. (<cite id="2f6wb"><a href="#zotero%7C20666258%2FKGH4AXV8">(Wei et al., 2022)</a></cite>) For example, LLMs can summarize texts, perform language translation, write working computer code, and compose informative responses on a wide array of subjects - all without specific training on how to perform such tasks. (<cite id="jq1bi"><a href="#zotero%7C20666258%2FWHCGSCI5">(Brown et al., 2020)</a></cite>) Moreover, these emergent capacities seem to "scale", meaning new models exhibit enhanced performance through training on ever-greater quantities of data and computation. (<cite id="1vggn"><a href="#zotero%7C20666258%2FRRUNZDAM">(Kaplan et al., 2020)</a></cite>) The nature of these emergent capacities remains a matter of intense research and debate, as do the ethical and legal questions surrounding their use. However, a growing body of research suggests that forms of historical competency are among these emergent capacities - but such competencies emerge unevenly, a reflection of the selective patterns of whose past gets is ultimately encoded into these models.

Assessing these capacities requires analytical humility, particularly regarding claims concerning LLM “knowledge” and “understanding.” Incisive critics of this technology characterize LLMs as “stochastic parrots” that excel at mimicry rather than comprehension. (<cite id="m0i3p"><a href="#zotero%7C20666258%2FMAEXPBX2">(Bender et al., n.d.)</a></cite>) Direct engagement with these models quickly reveals both their remarkable breadth and their narrow limits. Yet these limitations do not mean LLMs are of no interest or utility to historians. Indeed, the facility of these models in reflecting and recombining their training data represents a distinctive form of engaging with the past. Such engagement is grounded in probabilistic synthesis rather than analytical understanding, but precisely for that reason LLMs are worth investigating as a computational distillation of how the past has been recorded, preserved, and digitized. While the technical complexities of natural language processing and machine learning are increasingly common features of the historian's toolkit, technical expertise in these domains is helpful but not necessary to assess the historical capacities of LLMs. It is from the traditional vantage of source criticism that historians possess a unique perspective in evaluating these technologies. Indeed, the most widely-used measures for LLM historical knowledge was created not by computer scientists, but inadvertently by historians.

Benchmarking - that is systematically testing LLM performance against established forms of historical knowledge - provides an accessible methodology for both evaluating the capabilities of these models and the underlying patterns shaping these capacities. One of the most widely-used measures for LLM performance is the Massive Multitask Language Understanding (MMLU) benchmark, developed in 2021 by researchers led by Dan Hendryks. This benchmark contains nearly 16,000 questions from 57 academic disciplines ranging in difficulty from an elementary educational level to postgraduate curricula in professional domains like law and medicine. History is measured in this benchmark through some six hundred questions taken from the Advanced Placement (A.P.) curricula for U.S., European, and World history. (<cite id="k2mij"><a href="#zotero%7C20666258%2FQ269X8CB">(Hendrycks et al., 2021)</a></cite>) Hundreds of thousands of secondary students across the globe annually enroll in these curricula, which are designed to replicate the rigors of an introductory, university-level history course. The educators who developed and refined these programs likely never imagined their work would serve as a technical benchmark, and the appropriateness of such a standard can be debated. (<cite id="rhbf3"><a href="#zotero%7C20666258%2F8ISS2NP3">(Marshall, 2020)</a></cite>), (<cite id="3yqem"><a href="#zotero%7C20666258%2F7ERZCN5G">(Wong, 2018)</a></cite>) Yet this benchmark, however imperfect, offers historians an accessible means to evaluate this highly technical domain.

<!-- #endregion -->

<!-- #region editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} -->
In this benchmark, LLMs are given an excerpt from a historical source followed by a multiple-choice question, and are then instructed to identify the correct answer. Below is an example question drawn from the U.S. History curriculum:

**U.S. History Benchmark, Question 5:**

This question refers to the following information.

> “I was once a tool of oppression  
> And as green as a sucker could be  
> And monopolies banded together  
> To beat a poor hayseed like me.”  
>   
> “The railroads and old party bosses  
> Together did sweetly agree;  
> And they thought there would be little trouble  
> In working a hayseed like me. . . .”

*The Hayseed*

The song, and the movement that it was connected to, highlight which of the following developments in the broader society in the late 1800s?

**A**: Corruption in government, especially as it related to big business, energized the public to demand increased popular control and reform of local, state, and national governments.  
**B**: A large-scale movement of struggling African American and white farmers, as well as urban factory workers, was able to exert a great deal of leverage over federal legislation.  
**C**: The two-party system of the era broke down and led to the emergence of an additional major party that was able to win control of Congress within ten years of its founding.  
**D**: Continued skirmishes on the frontier in the 1890s with American Indians created a sense of fear and bitterness among western farmers.

**Correct Answer: A**


<!-- #endregion -->

<!-- #region citation-manager={"citations": {"b542w": [{"id": "20666258/Q269X8CB", "source": "zotero"}], "f6axv": [{"id": "20666258/BQPMNA6D", "source": "zotero"}], "gd63p": [{"id": "20666258/ZW8DJ3K3", "source": "zotero"}]}} jupyter={"outputs_hidden": false} -->
The MMLU benchmarks were first tested in 2021 against the then-leading LLM, OpenAI’s GPT-3. Twenty-five percent accuracy represented random chance; ninety percent performance reflected expert-level accuracy. GPT-3 achieved over fifty percent accuracy, and its performance in these fields numbered among the top third of all the academic disciplines in the benchmarks. However, in no field did GPT-3 achieve expert-level accuracy. (<cite id="b542w"><a href="#zotero%7C20666258%2FQ269X8CB">(Hendrycks et al., 2021)</a></cite>) Yet GPT-3's successors, scaled on ever greater quantities of data, moved decisively past the benchmark’s measurement for competence into the range of subject expertise.

Research into how LLMs develop capabilities during training - called mechanistic interpretability - offers suggestive evidence in explaining these gains. The Pythia LLM series, developed by EleutherAI, consists of models trained to different sizes, allowing researchers to study how different types of performance emerge during training. (<cite id="gd63p"><a href="#zotero%7C20666258%2FZW8DJ3K3">(Biderman et al., 2023)</a></cite>) Researchers developed an approach to track the emergence of historical capabilities across this series at distinct points in LLM development: across the model's training process, through its model architecture, and with increasing model size. They observed that physics and mathematics were domains that manifested earlier in the training process, while historical fluencies emerged only as the models grew larger in size. This research suggests that historical frameworks only emerge when a model has been trained on sufficient amounts of data, and that such capabilities require more sophisticated patterns built up during training. (<cite id="f6axv"><a href="#zotero%7C20666258%2FBQPMNA6D">(Sawmya et al., 2025)</a></cite>) However, once that scaling occurs and “emergence” is reached, the gains are substantial.

Rapid advances in model development have occurred since 2021, and so too has LLM performance on the MMLU. Below are results from a replication study conducted in September 2024 across a series of leading LLMs, along with the initial Hendryks test:
<!-- #endregion -->

```python jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["figure-llm-*"]
from IPython.display import Image
from IPython.display import display
from IPython.display import Markdown
metadata={
    "jdh": {
        "module": "object",
        "object": {
            "type":"image",
            "source": [
                "Accuracy of Selected LLMs on History Questions in the MMLU Benchmarks"
            ]
        }
    }
}

display(Image('./media/Table 1 - MMLU Benchmark Performance.png'), metadata=metadata)
```

<!-- #region citation-manager={"citations": {"lpqgi": [{"id": "20666258/35H7ZC5Z", "source": "zotero"}], "vf2y9": [{"id": "20666258/6HHA7DFR", "source": "zotero"}]}} editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["hermeneutics"] -->
The specific accuracy rates for GPT-3 for the initial Hendryks study: US History, 52.9%; European History, 53.9+%; and World History, 56.1%. Full data for questions for history and other disciplines can be found at: (<cite id="vf2y9"><a href="#zotero%7C20666258%2F6HHA7DFR">(Hendrycks, 2020/2023)</a></cite>)  Many thanks to Dan Hendrycks for sharing the discipline-specific accuracy rates for these fields. Data from this replication study can be accessed via the HELM Leaderboard for the MMLU Benchmark, hosted by the Center for Research on Foundation Models at Stanford University. (<cite id="lpqgi"><a href="#zotero%7C20666258%2F35H7ZC5Z">(Mai &#38; Liang, 2024)</a></cite>)
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"1fght": [{"id": "20666258/W4XUTNCR", "source": "zotero"}], "2ge3a": [{"id": "20666258/ENVC5ZIJ", "source": "zotero"}], "2su0n": [{"id": "20666258/WHCGSCI5", "source": "zotero"}], "3vfud": [{"id": "20666258/BRJE5S95", "source": "zotero"}], "3yijw": [{"id": "20666258/E38ZE6TS", "source": "zotero"}], "9d7mv": [{"id": "20666258/89T9PJV9", "source": "zotero"}], "bmavr": [{"id": "20666258/BERD6ARS", "source": "zotero"}], "ddwds": [{"id": "20666258/PVPPHZ56", "source": "zotero"}], "eh6ph": [{"id": "20666258/W569JM2K", "source": "zotero"}], "eyzvg": [{"id": "20666258/7MEV3F4T", "source": "zotero"}], "g7715": [{"id": "20666258/Z9B75488", "source": "zotero"}], "imrwi": [{"id": "20666258/GETAJ6CA", "source": "zotero"}], "m2rh9": [{"id": "20666258/84CCZGZA", "source": "zotero"}], "p832e": [{"id": "20666258/FCDIMVCZ", "source": "zotero"}], "q3drm": [{"id": "20666258/ZHJK8JPH", "source": "zotero"}], "rrwx6": [{"id": "20666258/ZKMMJTTK", "source": "zotero"}], "rsuve": [{"id": "20666258/FWUHDSFT", "source": "zotero"}], "ws34m": [{"id": "20666258/A6DI9F7T", "source": "zotero"}], "xmexl": [{"id": "20666258/P4D7WWG3", "source": "zotero"}], "xocqk": [{"id": "20666258/IDG87H2E", "source": "zotero"}], "yojtb": [{"id": "20666258/MAEXPBX2", "source": "zotero"}]}} editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} -->
Rapid advances on this benchmark have been made in just a few years, with a variety of commercial and open-source LLMs now demonstrating expert-level accuracy on all three of the history subject exams. These findings mirror the striking performance of generative AI models in other knowledge domains such as medical school curricula (<cite id="3vfud"><a href="#zotero%7C20666258%2FBRJE5S95">(Nori et al., 2023)</a></cite>), bar exams (<cite id="ddwds"><a href="#zotero%7C20666258%2FPVPPHZ56">(Katz, 2022)</a></cite>), and a host of other standardized assessments. (<cite id="xocqk"><a href="#zotero%7C20666258%2FIDG87H2E">(OpenAI, 2023)</a></cite>)

Yet, why do some LLMs perform better in some knowledge domains than others? Why does a model correctly answer one question, while other questions generate errors? There is a temptation to parse the model’s performance in ways relatable to our human perspective. The human test taker might approach the question by assessing what types of historical thinking is required, what sort of knowledge is offered by the options, and how the historical source relates to the question. But, of course, LLMs aren’t human - and unlike the human test taker, these models have already seen the questions in advance. In 2024 alone, over 950,000 students took A.P. History exams. (<cite id="p832e"><a href="#zotero%7C20666258%2FFCDIMVCZ">(College Board, n.d.)</a></cite>) Significant online resources have emerged to serve the sizable population of students and instructors participating in this international curriculum. Hundreds of exam questions have migrated online via the collective efforts of the test prep publishing industry, various study apps, and uploaded example tests. Given the scale of LLM training sets, many of these models have been trained on the very questions meant to test their competencies, a phenomenon known as “benchmark leakage.” (<cite id="1fght"><a href="#zotero%7C20666258%2FW4XUTNCR">(Xu et al., 2024)</a></cite>) This raises fundamental questions about what benchmarks like the MMLU actually measures. If those who critique LLMs as “stochastic parrots” are correct, these gains in performance come from improvements in models memorizing their training data through intensive computational training, and not through gains in analytical abilities. (<cite id="yojtb"><a href="#zotero%7C20666258%2FMAEXPBX2">(Bender et al., n.d.)</a></cite>) 

It’s useful then to reflect on what data the model is actually memorizing, as the emergent capabilities of LLMs derive from the vast datasets used to train them. The data collection built for training one of the first notable LLMs, OpenAI’s GPT-3 model, encompassed the majority of English-language Wikipedia, Reddit’s thousands of discussion forums, extensive corpora of digitized books and academic articles, and the billions of web pages contained in the Common Crawl repository. (<cite id="2su0n"><a href="#zotero%7C20666258%2FWHCGSCI5">(Brown et al., 2020)</a></cite>) The training sets used for subsequent LLMs often remain obscure, as AI firms keep their data a highly valued proprietary asset (ironically, perhaps, as the future of LLMs may depend on pending litigation concerning copyright infringement in the use of this data). But however vast the scale of such training sets, they nonetheless reflect the particular composition of our digital world: the most prevalent voices that have been digitized, the knowledge systems that inform them, and the dynamics of the analog world that produced them. Thus, such benchmarks might better be thought of as a measure of a LLM’s “artificial memory”, rather than “artificial intelligence.” 

Embedded within that memory are the legacies of the primary dataset for training LLMs: the Internet itself. While developers seek to remove potentially offensive texts from their training sets, the sheer scale of this data makes selective curation a significant technical challenge. This reality has troubled previous AI implementations. Well-intentioned researchers have created chatbots that spew hateful invective, human resources applications that refuse to hire female applicants, and algorithms based on criminal justice sentencing guidelines that starkly reinforce racial disparities already prevalent in the carceral system. (<cite id="rrwx6"><a href="#zotero%7C20666258%2FZKMMJTTK">(Barton, 2019)</a></cite>) Early models in the GPT series have been known to unexpectedly generate responses in innocuous contexts containing violent imagery, sexually explicit language, and racial, ethnic, and religious slurs. (<cite id="g7715"><a href="#zotero%7C20666258%2FZ9B75488">(Strickland, 2021)</a></cite>) These findings further confirm the prescient warnings offered by scholars such as Safiya Umoja Noble (<cite id="m2rh9"><a href="#zotero%7C20666258%2F84CCZGZA">(Noble, 2018)</a></cite>), Timnit Gebru (<cite id="bmavr"><a href="#zotero%7C20666258%2FBERD6ARS">(Gebru, 2020)</a></cite>), Ruha Benjamin (<cite id="xmexl"><a href="#zotero%7C20666258%2FP4D7WWG3">(Benjamin, 2019)</a></cite>), Kate Crawford (<cite id="eh6ph"><a href="#zotero%7C20666258%2FW569JM2K">(Crawford, 2021)</a></cite>),, and Trevor Paglen (<cite id="9d7mv"><a href="#zotero%7C20666258%2F89T9PJV9">(Crawford &#38; Paglen, n.d.)</a></cite>) on digital practices that reinforce analog inequalities. Some AI researchers consider such behaviors as lamentable but solvable problems through further technical advances, particularly with the use of methods like Reinforcement Learning from Human Feedback (RLHF). (<cite id="imrwi"><a href="#zotero%7C20666258%2FGETAJ6CA">(Christiano et al., 2023)</a></cite>) Reducing the impact of such biases is a significant research area, particularly through the creation of smaller, more carefully curated datasets for AI training. However, historians may share the skepticism of some researchers concerning such mitigations. (<cite id="2ge3a"><a href="#zotero%7C20666258%2FENVC5ZIJ">(Gehman et al., 2020)</a></cite>) Bias emerges from more than just explicit language or imagery but from the very structures of societies. Can any historical source be separated from its context as a neutral artifact, free of its creator’s perspective and the influences of its time? What about the untold millions of sources that make up the scale of an LLM’s training set?

A further consideration is not only do LLMs reproduce existing distortions from their source data, but also generate new forms of misinformation. LLMs tend to confidently assert error as fact, a phenomenon described by AI researchers as “hallucinations.” Unlike traditional historical sources where inconsistency or implausibility can be used to judge sources, hallucinations emerge from the same statistical processes that generate accurate information, errors seamlessly integrated and contextually appropriate even when factually false. This tendency represents a major challenge in LLM research and for many practical applications of this technology. (<cite id="q3drm"><a href="#zotero%7C20666258%2FZHJK8JPH">(Ji et al., 2023)</a></cite>) Rectifying such hallucinations is a significant area of LLM research. However, some scholars, like computational linguist Emily Bender, argue that such behaviors are inherent flaws in LLMs. (<cite id="eyzvg"><a href="#zotero%7C20666258%2F7MEV3F4T">(O’brien, 2023)</a></cite>)

Rather than viewing bias and hallucination as technical problems to solve, historians can use them to assess the digitized knowledge that shapes generative AI. LLMs reveal their limitations most clearly at the boundaries of their training data. Studying the nature of these errors offers valuable insights. In examining how historical capabilities emerge unevenly, historians can map more than just what LLMs know - they can also chart the gaps embedded within these models. Indeed, subsequent benchmarking studies help illustrate these patterns, exposing not just what LLMs “know”, but whose knowledge they amplify and whose perspectives remain obscured. Recent studies reveal that LLMs depend as much on the composition of training data as on computational scale, exposing systematic hierarchies across three key dimensions: linguistic representation, geographic inclusion, and interpretive complexity.

Linguistic hierarchies emerge starkly in specialized benchmarks. FoundaBench, a Chinese knowledge benchmark of over 3,000 questions akin to the MMLU, employed 200 multi-choice questions to assess LLM performance on questions on Chinese history gauged at the middle and high school levels. Smaller LLMs trained on extensive Chinese-language training data were found to outperform larger multilingual models like GPT-4. (<cite id="rsuve"><a href="#zotero%7C20666258%2FFWUHDSFT">(Li et al., 2024)</a></cite>) This finding suggests that training data composition, not just computational scale, is a key indicator of historical capabilities, revealing how linguistic representation creates uneven performance across cultural domains.

Geographic representation reveals even starker disparities. The HiST-LLM benchmark is based on 36,000 data points contained in the Seshat Global History Databank, a structured repository containing data from 600 historical societies from every United Nations region and drawn from the Neolithic period to the Industrial Revolution. Scholars used this benchmark to test seven LLMs on historical knowledge domains from across the globe. While all the LLMs performed better than random chance, none performed at an expert level. Moreover, LLM performance was weakest in questions relating to sub-Saharan Africa and Oceania. Such weaknesses are suggestive of the limited training data from these areas, reflecting a longer history of unequal access to digitalization, preservation, and infrastructure - digital divides now embedded in AI systems. (<cite id="3yijw"><a href="#zotero%7C20666258%2FE38ZE6TS">(Hauser et al., 2024)</a></cite>)

The most severe limitations emerge when benchmarks test interpretive rather than factual capabilities. The HiBenchLLM study challenged 14 leading LLMs on French regional history, specifically the history of the Poitou region. Unlike multiple-choice formats, these questions demanded responses featuring qualitative analysis and contextual understanding. Models averaged only 38% accuracy, with performance varying dramatically across question types. Even extensive French-language training proved insufficient for complex interpretive tasks, suggesting that LLMs are at their best when confronting “precise data about well-defined topics” but struggle with the analytical approaches central to historical scholarship. (<cite id="ws34m"><a href="#zotero%7C20666258%2FA6DI9F7T">(Chartier et al., 2025)</a></cite>)

Collectively, these benchmarking studies reveal LLMs as sources with distinct characteristics that directly inform their use in digital scholarship. When LLMs excel at certain tasks, they're often drawing on well-represented training patterns; when they falter, they may be encountering domains outside their inherited knowledge base. The uneven landscape of LLM historical knowledge - expert-level performance on A.P. curricula but systematic gaps in non-Western domains - exposes more than just technical limitations. These patterns reveal how historical knowledge has been preserved, prioritized, and made computationally accessible. LLMs do not know history; they instead reflect the particular version of history that has survived the transition from analog to digital, from local to global, and from a diverse chorus of individual voices to a singular, probabilistic synthesis. The following case studies examine this dynamic between amplification and silence across different use cases, beginning with technologies designed to capture voices themselves.
<!-- #endregion -->

<!-- #region editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} -->
## Case Studies: LLMs as Tools for Data Preparation
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"4lyzk": [{"id": "20666258/A427KHHQ", "source": "zotero"}], "of07d": [{"id": "20666258/ZSIP6FKE", "source": "zotero"}], "qefzr": [{"id": "20666258/7AQMGH6M", "source": "zotero"}]}} jupyter={"outputs_hidden": false} -->
Navigating the uneven terrain of LLM capabilities requires understanding what Ethan Mollick describes as the "jagged frontier": a landscape where peaks of performance in well-represented domains contrast sharply with valleys of limitation on the edges of their training data. (<cite id="qefzr"><a href="#zotero%7C20666258%2F7AQMGH6M">(Mollick, 2024)</a></cite>)  While future technical advances might smooth the contours of this frontier, assessing LLMs as sources can help scholars map both the peaks and valleys of this  knowledge space. Such approaches can aid historians in identifying both the emergent capacities of these technologies and their inherited limitations.

Understanding this terrain requires recognizing how LLM outputs can be guided and directed. One means for guidance is what researchers call “in-context learning,” an emergent ability enabling “language models to learn tasks given only a few examples in the form of demonstration.”  (<cite id="4lyzk"><a href="#zotero%7C20666258%2FA427KHHQ">(Dong et al., 2024)</a></cite>) This allows historians to selectively activate portions of the model's inherited knowledge across a spectrum of intervention: from guiding responses through contextual examples to retraining the model itself through fine-tuning on specialized datasets. However, such interventions come with their own costs and limitations. Simple prompting techniques require domain expertise but minimal technical resources, while fine-tuning demands substantial computational power and specialized datasets. Moreover, such approaches raise both practical and critical concerns: how best to instruct these models? What analytical frameworks do our demonstrations privilege? Do these interventions merely reshape the inherited biases of LLMs, rather than eliminate them?

The following case studies explore these dynamics at work by focusing on generative AI’s potential on foundational tasks instead of more complex forms of interpretation. This progression reflects a strategic approach to LLM evaluation: testing these systems first as tools for data preparation and cleanup, before considering their use as an analytical method. Digitized historical materials frequently require transcriptions, error correction, and metadata extraction - essential but time-consuming tasks that can become research bottlenecks. (<cite id="of07d"><a href="#zotero%7C20666258%2FZSIP6FKE">(Dasu &#38; Johnson, 2003)</a></cite>) By examining how LLMs can streamline these key tasks, historians can better assess both their utility and their systematic limitations. 

The applied case studies employ a variety of AI systems, most notably OpenAI's GPT-4o and Whisper. These choices merit explicit justification given the broader landscape of available AI systems. These models were selected for their consistent placement at the top of performance benchmarks, their widespread adoption in academic research, and their accessibility through user-friendly interfaces that require minimal technical expertise. This accessibility matters: historians seeking to evaluate AI capabilities do not need to navigate complex computational systems to begin their assessments. However, this use should not be read as an endorsement of proprietary AI systems. Throughout these case studies, open-source/open-weight alternatives like Meta's LLaMA series and EleutherAI's Pythia models are discussed alongside commercial offerings.

Rather than comprehensive performance assessments, these examples illustrate how source criticism can be used to understand generative AI’s potential as a research tool. The effectiveness of LLMs in even basic tasks depends heavily on the conventions and biases encoded in their training data. In mapping where these systems excel and where they struggle, these case studies identify methods for mitigating systematic limitations while leveraging the real strengths of LLMs. Indeed, scholars are already finding that within the confines of these limitations there is real potential to advance historical research.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} tags=["hermeneutics"] -->
Utilizing the Code Examples

The code demonstrations in this article require API keys to access AI models. These keys may offer free access for limited academic use and typically require only email registration. Availability may vary by region.

Below are instructions for obtaining keys from the two primary platforms used in these case studies:

For OpenAI's GPT-4o and Whisper:

1. Visit [https://platform.openai.com/signup](https://platform.openai.com/signup)
2. After email verification, go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
3. Click **"Create new secret key"** and copy the generated key
4. Insert your key in the code by replacing the empty quotes:
   os.environ["OPENAI_API_KEY"] = "your_key_here"

For HuggingFace (open-source models):

1. Create a free account at [https://huggingface.co/welcome](https://huggingface.co/welcome)  
2. Visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)  
3. Click **"Create new token"** 
4. Copy the generated token  
5. Insert your token in the code:
   os.environ["HF_TOKEN"] = "your_token_here"

Both platforms offer extensive documentation for troubleshooting. Readers preferring to avoid commercial services can focus exclusively on the HuggingFace examples, which use open-source alternatives.
<!-- #endregion -->

<!-- #region editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} -->
## Whose Voices Are Heard? LLMs and the Hierarchies of Oral History
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"8y2sx": [{"id": "20666258/DM9N78FC", "source": "zotero"}], "fllrq": [{"id": "20666258/66HNW5BJ", "source": "zotero"}], "h0o74": [], "l22zs": [{"id": "20666258/9N7V3WDP", "source": "zotero"}], "md1oh": [], "ziths": [{"id": "20666258/2KU6Q2RE", "source": "zotero"}]}} editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} -->
Oral history provides a useful case study for demonstrating the potential utility and risks of generative AI. Transcribing audio recordings is a central activity in this methodology, a task typically requiring significant time and labor. As one oral history guide notes, transcribing a single hour typically requires six to eight hours of manual processing and review. (<cite id="h0o74"><a href="#zotero%7C20666258%2FM64T96PV">(Ritchie, 2003)</a></cite>) However, specialized generative AI models enable significant streamlining of this task, potentially transforming hours of labor into just minutes. Scholars are already using these techniques to complete multi-lingual transcriptions of aging and vulnerable media (<cite id="8y2sx"><a href="#zotero%7C20666258%2FDM9N78FC">(Lehečka et al., 2023)</a></cite>) and powering new forms of community-based scholarship and teaching. (<cite id="md1oh"><a href="#zotero%7C20666258%2F2NNI9P9W">(Rochester Institute of Technology, n.d.)</a></cite>) 
However, like an audio system calibrated for certain frequencies, these models perform best on speech most prominently featured in their training. Underrepresented voices and linguistic varieties may be lost in the static.

Notable among generative AI models is Whisper, an audio transcription and translation model developed by OpenAI that belongs to the same Transformer family as the GPT LLM series. Trained on over 680,000 hours of audio recordings and paired transcripts, this model demonstrates performance comparable (and in some cases exceeding) the accuracy rates of human transcriptions on performance benchmarks. Despite these strengths, Whisper is prone to error and hallucination, like other forms of generative AI. (<cite id="l22zs"><a href="#zotero%7C20666258%2F9N7V3WDP">(Koenecke et al., 2024)</a></cite>) Most critically, Whisper reproduces the same linguistic hierarchies found in text-based LLMs.  Approximately 65% of the training data is in English (some 438,000 hours of audio), while the remaining 35% of the corpus represents 98 other distinct languages. The scale of coverage in this subset varies significantly, with many languages represented by less than a thousand hours of data. (<cite id="fllrq"><a href="#zotero%7C20666258%2F66HNW5BJ">(Radford et al., 2022)</a></cite>) Such disparities reflect and reinforce digital divides, creating a computational hierarchy where a privileged few languages become readily audible to AI models while many others remain indistinct.

In this test we’ll examine how Whisper’s training impacts its performance on oral history transcriptions in two different languages. Below are the first two minutes of a transcribed oral history of historian John Hope Franklin by the Southern Oral History Program at the University of North Carolina. (<cite id="ziths"><a href="#zotero%7C20666258%2F2KU6Q2RE">(Franklin, 1990)</a></cite>) Recorded on audiotape in 1990, this segment features multiple voices, crosstalk, filler words, and background noise - typical features for many oral history recordings, but features that nonetheless complicate efforts to create accurate transcripts. In the code below, we will use Whisper to transcribe the audio segment and compare it against the professionally prepared transcript.
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"8w84i": [{"id": "20666258/K7JFQAA5", "source": "zotero"}]}} editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["hermeneutics"] -->
The Whisper series is offered as a series of voice recognition and voice translation models across several tiers of computing power and available on sites like HuggingFace. However, for simplicity this demonstration code uses OpenAI’s API for the Whisper-2-large model. As of September 2024, OpenAI charged $0.36 per hour of recorded time for transcriptions using the API.

For a detailed and informative tutorial on using and analyzing Whisper, see: (<cite id="8w84i"><a href="#zotero%7C20666258%2FK7JFQAA5">(Schultz, 2024)</a></cite>)

<!-- #endregion -->

```python jupyter={"outputs_hidden": false}
# installing libraries
!pip install openai
!pip install jiwer
!pip install rich
!pip install huggingface-hub
```

```python editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""}
# Enter OpenAI API key in the space below.
# Access to OpenAI's API keys can be found here: https://beta.openai.com/signup

import os
os.environ["OPENAI_API_KEY"] = " "
```

```python editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""}
# Code for transcribing oral history segment with Whisper API

import requests
from openai import OpenAI
import time
from rich.console import Console
from rich.console import Console
from rich.text import Text

# Initialize the OpenAI client
client = OpenAI()

# URL for the audio file on GitHub
audio_url = "https://github.com/jdh-observer/JZx9gw7iwGxb/raw/refs/heads/main/media/A-0339_edited.mp3"

# Save location for the downloaded audio file
file_path = "./A-0339_edited.mp3"

# Download the audio file and save it locally
response = requests.get(audio_url)
with open(file_path, "wb") as audio_file:
    audio_file.write(response.content)

# Measure the transcription time for the audio file
start_time = time.time()

# Transcribe the audio
with open(file_path, "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file
    )
whisper_transcript = transcription.text

end_time = time.time()

# Calculate the actual transcription time
automation_time = end_time - start_time

# Calculate the estimated transcription time for 1 hour based on the transcription time for audio segment
audio_length_seconds = 153  # 2 minutes and 33 seconds
estimated_time_for_one_hour = (automation_time / audio_length_seconds) * 3600  # Time for 1 hour (3600 seconds)

# Convert estimated time for better readability
hours = int(estimated_time_for_one_hour // 3600)
minutes = int((estimated_time_for_one_hour % 3600) // 60)
seconds = int(estimated_time_for_one_hour % 60)

console = Console()

# Outputs with rich formatting
output_text = (
    f"[bold]Whisper Transcription time:[/bold] {automation_time:.2f} seconds\n\n"
    f"[bold]Estimated Transcription Time for an hour recording at this rate:[/bold] "
    f"{hours} hours, {minutes} minutes, {seconds} seconds\n\n"
    f"[bold]Raw Whisper Transcript[/bold]\n"
    f"[dim]{whisper_transcript}[/dim]"
)

# Print outputs
console.print(output_text, width=console.size.width)
```

<!-- #region jupyter={"outputs_hidden": false} -->
The code below generates a audio player to listen to the audio segment. Listen and follow along to observe Whisper’s performance.
<!-- #endregion -->

```python jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["sound-franklin-*"]
from IPython.display import Audio
# URL for the audio file on GitHub
audio_url = "https://github.com/jdh-observer/JZx9gw7iwGxb/raw/refs/heads/main/media/A-0339_edited.mp3"

# Displaying citation
citation_text = (
    "**Citation:**\n"
    "*“John Hope Franklin and John Egerton, Conducted by Oral History Interview with John Hope Franklin, "
    "July 27, 1990. Interview A-0339. Southern Oral History Program Collection (#4007).”*\n"
    "https://docsouth.unc.edu/sohp/A-0339/menu.html"
)
metadata = {
    "jdh": {
        "object": {
            "type": "image",
            "source": [
                citation_text
            ]
        }
    }
}
# Load and play the saved audio file
display(Audio(audio_url), metadata=metadata)
```

<!-- #region jupyter={"outputs_hidden": false} -->
Based on the professional standard, this excerpt would take approximately fifteen to twenty minutes to manually transcribe. Whisper achieved this in less than ten seconds.

How accurate is the model compared to a human-produced transcript? Due to the stochastic nature of these models, each time you run this code slightly different variations might occur, particularly in the most challenging segments. The code block below visualizes a sample transcription produced by Whisper that was annotated and compared against the original. Notable omissions and discrepancies are highlighted. Whisper’s accuracy is then calculated via a standard metric for audio transcription, the word error rate (WER).
<!-- #endregion -->

```python jupyter={"outputs_hidden": false}
import requests
import re
from jiwer import wer
from IPython.display import display, HTML

# Function to clean HTML tags
def clean_html(text):
    return re.sub(r'<.*?>', '', text)

# URLs for the transcripts on GitHub
original_file_url = "https://raw.githubusercontent.com/jdh-observer/JZx9gw7iwGxb/refs/heads/main/media/revised_original_transcript_formatted.txt"
whisper_file_url = "https://raw.githubusercontent.com/jdh-observer/JZx9gw7iwGxb/refs/heads/main/media/revised_whisper_transcript_formatted.txt"

# Download and read the contents of the original and whisper transcripts
original_transcript = requests.get(original_file_url).text
whisper_transcript = requests.get(whisper_file_url).text

# Clean the transcripts for WER calculation
cleaned_original_transcript = clean_html(original_transcript)
cleaned_whisper_transcript = clean_html(whisper_transcript)

# Calculate the Word Error Rate (WER)
error_rate = wer(cleaned_original_transcript, cleaned_whisper_transcript)

# Add <br> tags to preserve line breaks in the text
original_transcript = original_transcript.replace('\n', '<br>')
whisper_transcript = whisper_transcript.replace('\n', '<br>')

# Ensure that color highlighting also includes bolding
whisper_transcript = whisper_transcript.replace(
    'style="background-color: #fbb;"',
    'style="background-color: #fbb; font-weight: bold;"'
)

original_transcript = original_transcript.replace(
    'style="background-color: #bfb;"',
    'style="background-color: #bfb; font-weight: bold;"'
)

# Display the two transcripts side by side using HTML in Jupyter
html_content = f'''
<div style="display: flex;">
    <div style="width: 50%; padding-right: 20px; border-right: 1px solid black;">
        <h4>Original Transcript: (discrepancies in green)</h4>
       {original_transcript}
    </div>
    <div style="width: 50%; padding-left: 20px;">
        <h4>Whisper Transcript: (discrepancies in red)</h4>
        {whisper_transcript}
    </div>
</div>
<br><br>
<div style="text-align: center;">
    <h4>Word Error Rate (WER) for Whisper: {error_rate:.2%}</h4>
</div>
'''

# Render the HTML content in Jupyter
display(HTML(html_content))
```

<!-- #region citation-manager={"citations": {"8go9h": [{"id": "20666258/C5683JU6", "source": "zotero"}], "rg3e9": [{"id": "20666258/JIJPRUQN", "source": "zotero"}]}} jupyter={"outputs_hidden": false} -->
There are some suggestive observations we can take from these results. Closer inspection of the Whisper transcript shows some errors, a significant omission, differences in syntax, and literal transcriptions that an edited transcript would likely leave out. But given the media format and its audio quality, the oral historian has a solid first draft in seconds. While the WER score indicates a need for final human review, that review will take considerably less effort and enable oral historians to shift their focus to interpreting, annotating, and validating their transcriptions. And even the best human transcriptions still contain errors. Take note of the final paragraph in the original transcript, which names Harvard as the destination of E. Franklin Frazier in 1934; but the noted sociologist actually joined the faculty of Howard University. Here Whisper accurately corrects a human error in the transcription. Whisper’s performance in this particular recording offers an example of its comparative strength in standard forms of English, reflecting the weighting of its training process.

However, its reliability suffers when encountering languages less represented in its training. In the second test we’ll use a segment of an oral history conducted in Vietnamese, a language representing some 2,300 hours of the Whisper dataset, or less than one percent of the model’s total training data. (<cite id="8go9h"><a href="#zotero%7C20666258%2FC5683JU6">(Radford et al., 2021)</a></cite>) Below is an annotated interview of Bao Ninh conducted by the staff of the Vietnam Center and Sam Johnson Vietnam Archive at Texas Tech University versus Whisper’s output. (<cite id="rg3e9"><a href="#zotero%7C20666258%2FJIJPRUQN">(Ninh, 2005)</a></cite>)
<!-- #endregion -->

```python jupyter={"outputs_hidden": false}
# Code for transcribing oral history segment with Whisper API

import requests
from openai import OpenAI
import time
from rich.console import Console
from rich.console import Console
from rich.text import Text

# Initialize the OpenAI client
client = OpenAI()

# URL for the audio file on GitHub
audio_url = "https://github.com/jdh-observer/JZx9gw7iwGxb/raw/refs/heads/main/media/OH0435_edited.mp3"

# Save location for the downloaded audio file
file_path = "./OH0435_edited.mp3"

# Download the audio file and save it locally
response = requests.get(audio_url)
with open(file_path, "wb") as audio_file:
    audio_file.write(response.content)

# Measure the transcription time for the audio file
start_time = time.time()

# Transcribe the audio
with open(file_path, "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file
    )
whisper_transcript = transcription.text

end_time = time.time()

# Calculate the actual transcription time
automation_time = end_time - start_time

# Calculate the estimated transcription time for 1 hour based on the transcription time for audio segment
audio_length_seconds = 170  # 2 minutes and 50 seconds 
estimated_time_for_one_hour = (automation_time / audio_length_seconds) * 3600  # Time for 1 hour (3600 seconds)

# Convert estimated time for better readability
hours = int(estimated_time_for_one_hour // 3600)
minutes = int((estimated_time_for_one_hour % 3600) // 60)
seconds = int(estimated_time_for_one_hour % 60)

console = Console()

# Outputs with rich formatting
output_text = (
    f"[bold]Whisper Transcription time:[/bold] {automation_time:.2f} seconds\n\n"
    f"[bold]Estimated Transcription Time for an hour recording at this rate:[/bold] "
    f"{hours} hours, {minutes} minutes, {seconds} seconds\n\n"
    f"[bold]Raw Whisper Transcript[/bold]\n"
    f"[dim]{whisper_transcript}[/dim]"
)

# Print outputs
console.print(output_text, width=console.size.width)
```

<!-- #region jupyter={"outputs_hidden": false} -->
Below is the raw audio of the interview. As in the last recording, there is crosstalk, background noise, and multiple speakers. 
<!-- #endregion -->

```python jupyter={"outputs_hidden": false} tags=["sound-baoninh-*"]
from IPython.display import Audio
# URL for the audio file on GitHub
audio_url = "https://github.com/jdh-observer/JZx9gw7iwGxb/raw/refs/heads/main/media/OH0435_edited.mp3"

# Displaying citation
citation_text = (
    "**Citation:**\n"
    "*“Interview with Bao Ninh, Conducted by Richard B. Verrone and Khanh Le. "
    "March 17, 2005. Interview OH0435. Vietnam Center and Sam Johnson Vietnam Archive, Texas Tech University.”*\n"
    "https://vva.vietnam.ttu.edu/repositories/2/digital_objects/412347"
)
metadata = {
    "jdh": {
        "object": {
            "type": "image",
            "source": [
                citation_text
            ]
        }
    }
}
# Load and play the saved audio file
display(Audio(audio_url), metadata=metadata)

```

<!-- #region jupyter={"outputs_hidden": false} -->
To measure Whisper's accuracy another annotated transcription is compared against the professionally prepared transcript. Notable omissions and discrepancies are highlighted, with the WER score displayed. 
<!-- #endregion -->

```python jupyter={"outputs_hidden": false}
import requests
import re
from jiwer import wer
from IPython.display import display, HTML

# Function to clean HTML tags
def clean_html(text):
    return re.sub(r'<.*?>', '', text)

# URLs for the transcripts on GitHub
original_file_url = "https://raw.githubusercontent.com/jdh-observer/JZx9gw7iwGxb/refs/heads/main/media/vi_original_highlighted_2.txt"
whisper_file_url = "https://raw.githubusercontent.com/jdh-observer/JZx9gw7iwGxb/refs/heads/main/media/vi_whisper_highlighted_2.txt"

# Download and read the contents of the original and whisper transcripts
original_transcript = requests.get(original_file_url).text
whisper_transcript = requests.get(whisper_file_url).text

# Clean the transcripts for WER calculation
cleaned_original_transcript = clean_html(original_transcript)
cleaned_whisper_transcript = clean_html(whisper_transcript)

# Calculate the Word Error Rate (WER)
error_rate = wer(cleaned_original_transcript, cleaned_whisper_transcript)

# Add <br> tags to preserve line breaks in the text
original_transcript = original_transcript.replace('\n', '<br>')
whisper_transcript = whisper_transcript.replace('\n', '<br>')

# Ensure that color highlighting also includes bolding
whisper_transcript = whisper_transcript.replace(
    'style="background-color: #fbb;"',
    'style="background-color: #fbb; font-weight: bold;"'
)

original_transcript = original_transcript.replace(
    'style="background-color: #bfb;"',
    'style="background-color: #bfb; font-weight: bold;"'
)

# Display the two transcripts side by side using HTML in Jupyter
html_content = f'''
<div style="display: flex;">
    <div style="width: 50%; padding-right: 20px; border-right: 1px solid black;">
        <h4>Original Transcript: (discrepancies in green)</h4>
       {original_transcript}
    </div>
    <div style="width: 50%; padding-left: 20px;">
        <h4>Whisper Transcript: (discrepancies in red)</h4>
        {whisper_transcript}
    </div>
</div>
<br><br>
<div style="text-align: center;">
    <h4>Word Error Rate (WER) for Whisper: {error_rate:.2%}</h4>
</div>
'''

# Render the HTML content in Jupyter
display(HTML(html_content))
```

<!-- #region citation-manager={"citations": {"cjzxc": [{"id": "20666258/5UK7S2IS", "source": "zotero"}], "oiqad": [{"id": "20666258/66HNW5BJ", "source": "zotero"}], "sivzt": [{"id": "20666258/DM9N78FC", "source": "zotero"}]}} jupyter={"outputs_hidden": false} -->
These results offer a marked contrast to the previous example. In the English language transcript, Whisper largely captured the audio content with reasonable fidelity, although its errors and omissions still indicate the importance of human review. In contrast, the Vietnamese transcript features a variety of significant distortions, ranging from phonetic errors to hallucinated text. Perhaps most concerning is the model’s inability to effectively capture important terms from the recording such as Việt Minh (North Vietnam’s Communist movement) and Tây Nguyên (the Central Highlands region of Vietnam). Such findings illustrate how generative AI's performance degrades at the margins of its training.

However, the relationship between training data and performance is more nuanced than simple hour counts suggest, as the diversity of data proves as crucial as its scale. For example, a study examining Whisper’s performance on two Mandarin language datasets found a wide variation in transcription reliability, despite the presence of 35,000 hours of this language in Whisper’s dataset, making it the second most prominent language besides English. (<cite id="oiqad"><a href="#zotero%7C20666258%2F66HNW5BJ">(Radford et al., 2022)</a></cite>) Closer examination revealed that one of the datasets was primarily based on mainland Chinese dialects while the other was compiled from Mandarin speakers from Singapore and Malaysia. While the underlying language of both datasets are the same, the regional distinctions between the two resulted in a significant divergence in performance. (<cite id="cjzxc"><a href="#zotero%7C20666258%2F5UK7S2IS">(Peng et al., 2023)</a></cite>) Such findings suggest that data hierarchies operate not just between languages but within them, reflective of the larger geographic and cultural patterns encoded within generative AI systems.

Understanding these limitations enables targeted strategies for overcoming them. One notable example is the development of custom transcription models trained for a distinctive archive. The Shoah Foundation Institute’s Visual History Archive (SHI-VHA) is a collection of 54,000 interviews of Holocaust survivors in 32 languages. While the bulk of these recordings are in English, a substantial collection were recorded in German and Czech. Moreover, the advanced age of most of the interviewees (averaging 75 years old), the presence of heavy accents, and the emotionally charged nature of the interviews resulted in recordings difficult for multilingual models like Whisper to accurately transcribe. Researchers at the University of West Bohemia Pilsen developed custom transcription models trained on the distinct features of the archive itself, and specialized for each language. Collectively these monolingual models were far smaller in terms of their datasets (24,000 hours in total) when compared the scale of Whisper’s training (680,000 hours), but that specialization made all the difference. Fine-tuning these specialized models produced transcriptions that significantly outperformed Whisper. (<cite id="sivzt"><a href="#zotero%7C20666258%2FDM9N78FC">(Lehečka et al., 2023)</a></cite>)

What are the implications of these findings? In considering generative AI models like Whisper for oral history transcriptions, historians should consider whose voices are most amplified by these technologies and whose voices are comparatively silent. For interviews conducted in high-frequency languages and in standard diction, such models can effectively streamline and accelerate a labor-intensive process, even if the final output still requires human review. Yet even in well-represented languages the influence of dialects, accents, and other demographic factors can play significant roles in the fidelity of the transcription. For historians working in lower frequency languages, generative AI models may prove particularly problematic. These shortcomings can be addressed through the creation of custom models trained on specialized datasets reflective of their targeted domain. However, such approaches possess their own challenges. Assembling datasets of sufficient size and diversity is itself a labor intensive process, and the computational resources and technical knowledge required for such training is often significant.   

Yet not all customized mitigations require substantial resources or technical expertise. The capacity of generative AI models for in-context learning can be leveraged through carefully designed prompts and representative examples. This approach proves particularly valuable when applied to another foundational challenge in digital history: correcting errors in optical character recognition scans.

<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} -->
## Whose Past Becomes Legible? OCR Correction and the Limits of Computational Vision
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"9gega": [{"id": "20666258/9QVGGFHF", "source": "zotero"}], "xedxa": [{"id": "20666258/V49364MS", "source": "zotero"}]}} editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} -->
Another use case for generative AI models is error correction of optical character recognition (OCR) scans. To make historical sources useful for digital analysis they must first be transformed into a computationally legible format, converting documents into searchable text. While OCR techniques are effective when applied to well-preserved documents using modern typefaces, such approaches falter when applied to the diverse texts often encountered by historians. OCR scans of handwritten manuscripts, documents with complex layouts and historical typefaces, and materials degraded by the passage of time often contain significant errors. As Ian Milligan has demonstrated, the prevalence of such errors creates an "illusionary order" where digitized historical collections appear comprehensively accessible while actually containing systematic gaps. (<cite id="9gega"><a href="#zotero%7C20666258%2F9QVGGFHF">(Milligan, 2013)</a></cite>) LLMs are potential tools in closing these gaps. However, these models are encoded with temporal and linguistic hierarchies inherited from their training data, making historical sources unevenly legible to AI systems. LLM-based OCR correction thus risks creating its own illusionary order, where strengths in well-represented domains mask systematic weaknesses.

To understand these risks, it's useful to compare LLMs to existing OCR approaches widely used in digital history. Machine learning techniques, such as those pioneered by the research team at Transkribus, have enhanced the quality, speed, and cost-effectiveness of OCR for a broad range of historical texts. (<cite id="xedxa"><a href="#zotero%7C20666258%2FV49364MS">(Muehlberger et al., 2019)</a></cite>) However, even these approaches require assembling datasets of annotated examples to train specialized models. While such domain-specific training yields measurable improvements, the resources needed to build these models and their narrow applicability limit their scalability. LLMs, in contrast, have been trained on vast datasets of varying formats, languages, and styles. Researchers are exploring the suitability of these models as a generalized method for OCR correction. As in other contexts, LLM performance in these areas offers insights into the historically contingent nature of their training.

The technical frontier in this domain is particularly jagged. LLMs demonstrate promising results for error correction in areas that are well-reflected in their underlying training data, particularly multimodal models trained on both text and images. Moreover, “prompt engineering” - the use of tailored prompts to leverage LLMs’ capacities for in-context learning - offers a flexible approach to adapting corrections to specific formats and styles. However, despite this potential LLMs demonstrate considerable variance in performance based on the quality of the underlying training data and model's size. LLM-based OCR correction can achieve significant improvements for high-quality images and well-represented languages, but for poor-quality images and underrepresented languages LLMs can actually degrade the underlying accuracy of OCR scans.

The following examples demonstrate this variance. The image below is taken from a newspaper published in a German prisoner-of-war camp in the United States during World War II and later microfilmed by the Library of Congress. POW editors used American typesetting equipment in producing this publication, and the text lacks umlauts, eszetts, or other diacritics used in German language sources of the era. Such omissions give these texts a unique textual “accent” that helps illustrate LLM capacities for OCR correction.
<!-- #endregion -->

```python editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""}
from PIL import Image
from IPython.display import display, Image as IPImage

# Load and resize the image
ocr2_url = 'https://raw.githubusercontent.com/jdh-observer/JZx9gw7iwGxb/refs/heads/main/media/die_lotse_6-30-45_1.png'
image2 = Image.open('media/die_lotse_6-30-45_1.png')

# Set new dimensions for the resized image
new_width = 600
new_height = int(image2.height * (new_width / image2.width))

# Resize the image for better visualization
resized_image = image2.resize((new_width, new_height), Image.LANCZOS)

# Convert the PIL image to a format compatible with IPython display 
resized_image.save("/tmp/resized_image.png")  
# Format the citation text using rich
citation_text = (
    "**Source:** *“Nur ein Film?,”* *Die Lotse* (Camp McCain, Mississippi), 30 June 1945.\n"
    "In: Karl John Richard Arndt, editor. *German P.O.W. Camp Papers*. (Washington, D.C.: Library of Congress Photoduplication Service, 1965). Reel 9."
)
metadata = {
    "jdh": {
        "object": {
            "type": "image",
            "source": [
                citation_text
            ]
        }
    }
}
display(IPImage(filename="/tmp/resized_image.png"), metadata=metadata) 
```

<!-- #region jupyter={"outputs_hidden": false} -->
The following code block demonstrates an LLM’s facility in correcting a raw OCR scan of this image. Here an initial scan produced by Google’s Cloud Vision OCR Service is fed to a small LLM, LLaMA-3.2-3B-Instruct, a model developed by Meta. The model’s prompt instructions are simple: “Correct this OCR output. Provide only the corrected text.”  By again using the word error rate (WER), we can compare the relative accuracy of the model against human performance. Discrepancies between the human transcription and the LLM outputs are highlighted.
<!-- #endregion -->

```python editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""}
# Set your HF API token here 
os.environ["HF_TOKEN"] = ""

# Initialize Hugging Face Inference Client via Inference Endpoint
client = InferenceClient(
    provider="auto",
    api_key=os.environ["HF_TOKEN"]
)
```

```python jupyter={"outputs_hidden": false}
# Functions for Comparing and Displaying OCR Corrections

import requests
import difflib
from jiwer import wer
from IPython.display import display, HTML

# Fetch text content from a URL
def fetch_text_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text.strip()

# Annotate diffs between two texts
def annotate_differences(diff, target):
    result = []
    for word in diff:
        if word.startswith('+') and target == 'ocr':
            result.append(f'<span style="color:green;background-color:#e6ffe6;">{word[2:]}</span>')
        elif word.startswith('-') and target == 'human':
            result.append(f'<span style="color:red;background-color:#ffe6e6;">{word[2:]}</span>')
        elif word.startswith(' '):
            result.append(word[2:])
    return ' '.join(result)

# Display WER and annotated HTML
def display_side_by_side(ocr_output, corrected_output, title_ocr, title_corrected):
    diff = list(difflib.Differ().compare(ocr_output.split(), corrected_output.split()))
    ocr_annotated = annotate_differences(diff, 'ocr')
    corrected_annotated = annotate_differences(diff, 'human')
    html = f'''
    <div style="display: flex;">
        <div style="width: 50%; padding-right: 20px; border-right: 1px solid black;">
            <h4>{title_ocr}</h4><div>{ocr_annotated}</div>
        </div>
        <div style="width: 50%; padding-left: 20px;">
            <h4>{title_corrected}</h4><div>{corrected_annotated}</div>
        </div>
    </div>'''
    display(HTML(html))

def run_comparison(ocr_output, corrected_output, title_ocr, title_corrected):
    display_side_by_side(ocr_output, corrected_output, title_ocr, title_corrected)
    error_rate = wer(corrected_output, ocr_output)
    label = title_corrected.split('(')[0].strip()
    display(HTML(f'<h4>Word Error Rate (WER) for {label}: {error_rate:.2%}</h4>'))

```

```python jupyter={"outputs_hidden": false}
# OCR Correction with LLaMA-3.2-3B-Instruct, simple prompt

from huggingface_hub import InferenceClient

# Initialize Hugging Face client (assumes HF_TOKEN in env)
client = InferenceClient()

# Load files
ocr_output = fetch_text_from_url("https://raw.githubusercontent.com/jdh-observer/JZx9gw7iwGxb/refs/heads/main/media/die_lotse_1_ocr_output.txt")
human_output = fetch_text_from_url(https://raw.githubusercontent.com/jdh-observer/JZx9gw7iwGxb/refs/heads/main/media/die_lotse_1_human_correction.txt")
prompt_instructions = "Correct this OCR output. Provide only the corrected text."

# Query LLaMA
def query_llama(prompt, text):
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[{"role": "user", "content": f"{prompt_instructions}\n\n{text}"}]
    )
    return response.choices[0].message.content

llama_output = query_llama(ocr_prompt, ocr_output)

# Display comparison
run_comparison(
    ocr_output, human_output,
    "Human Corrected Transcript (Corrections in green)",
    "OCR Transcript (Errors in red)"
)

run_comparison(
    llama_output, human_output,
    "Human Corrected Transcript (Corrections in green)",
    "LLaMA Corrected Transcript (Errors in red)"
)

```

<!-- #region citation-manager={"citations": {"fdhyl": [{"id": "20666258/BIZ89DQ7", "source": "zotero"}]}} editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} -->
Here we can observe the limitations of a smaller model and a simple prompting approach when applied to a novel form of historical source material. Instead of correcting the scan, the model actually makes the output less accurate. Moreover, the model adds German diacritics and special characters. While such characters would be expected in a probabilistic correction, they do not actually reflect the original text, resulting in a distortion of the underlying source. This failure illustrates a fundamental challenge: LLMs trained on modern digital text tend to struggle with historical documents that diverge from contemporary norms. This pattern of anachronistic standardization extends far beyond this example. A recent study of OCR corrections of German medical periodicals from 1951-1990 reveals that LLMs systematically impose newer orthographic standards from the post-1996 language reforms onto earlier texts. (<cite id="fdhyl"><a href="#zotero%7C20666258%2FBIZ89DQ7">(Danilova &#38; Aangenendt, 2025)</a></cite>) LLM training data dominated by contemporary web content thus creates temporal hierarchies within languages. Older standards, historical typography, and period-specific language render certain historical sources more opaque to AI models.

However, these defects can be partially mitigated by providing the model with more detailed guidance. In the next example, the simple instruction to  "correct this OCR output" is replaced with a [prompt](https://raw.githubusercontent.com/jdh-observer/JZx9gw7iwGxb/refs/heads/main/media/prompts/ocr_prompt.txt) containing three key elements: clear instructions about the specific OCR correction task, relevant context about the source material, and several representative examples demonstrating how the model should address similar corrections. These examples demonstrate the desired correction style while explicitly instructing the model to preserve the text's original character - including the absence of German diacritics like “ü” and “ß” that would normally appear in German text but were omitted due to the wartime printing constraints. This approach, called few-shot prompting, leverages LLMs' capacity for in-context learning and adapting behavior based on demonstration. By showing the model several examples of appropriate corrections, historians can guide AI performance toward domain-specific requirements without requiring technical interventions.
<!-- #endregion -->

```python editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""}
# OCR Correction with LLaMA-3.2-3B-Instruct, few-shot prompting

from huggingface_hub import InferenceClient

# Initialize Hugging Face client (assumes HF_TOKEN in env)
client = InferenceClient()

# Load files
ocr_output = fetch_text_from_url("https://raw.githubusercontent.com/jdh-observer/JZx9gw7iwGxb/refs/heads/main/media/die_lotse_1_ocr_output.txt")
human_output = fetch_text_from_url("hhttps://raw.githubusercontent.com/jdh-observer/JZx9gw7iwGxb/refs/heads/main/media/die_lotse_1_human_correction.txt")
ocr_prompt = fetch_text_from_url("https://raw.githubusercontent.com/jdh-observer/JZx9gw7iwGxb/refs/heads/main/media/prompts/ocr_prompt.txt")

# Query LLaMA
def query_llama(prompt, text):
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[{"role": "user", "content": f"{prompt}\n\n{text}"}]
    )
    return response.choices[0].message.content

llama_output = query_llama(ocr_prompt, ocr_output)

# Display comparison
run_comparison(
    ocr_output, human_output,
    "Human Corrected Transcript (Corrections in green)",
    "OCR Transcript (Errors in red)"
)

run_comparison(
    llama_output, human_output,
    "Human Corrected Transcript (Corrections in green)",
    "LLaMA Corrected Transcript (Errors in red)"
)

```

<!-- #region jupyter={"outputs_hidden": false} -->
In using few-shot prompting the model now actually demonstrates improvement in the accuracy of the OCR scan, although significant errors remain uncorrected and diacritics are still added in the output. While the prompt proved useful in steering the model towards greater accuracy, the effectiveness of this approach scales with the size of the model. Indeed, substantial improvements are produced when applying this prompt to a LLM an order of magnitude in size, OpenAI’s GPT-4o. Trained on a vaster corpus of multilingual text and possessing more robust capacities for in-context learning, it demonstrates both greater accuracy in correcting the OCR scan and greater fidelity in adhering to the prompt instructions.
<!-- #endregion -->

```python jupyter={"outputs_hidden": false}
# OCR correction with GPT-4o, few shot prompting

from openai import OpenAI

# Initialize client (assumes OPENAI_API_KEY already set in env)
client = OpenAI()

# Load files
ocr_output = fetch_text_from_url("https://raw.githubusercontent.com/jdh-observer/JZx9gw7iwGxb/refs/heads/main/media/die_lotse_1_ocr_output.txt")
human_output = fetch_text_from_url("https://raw.githubusercontent.com/jdh-observer/JZx9gw7iwGxb/refs/heads/main/media/die_lotse_1_human_correction.txt")
ocr_prompt = fetch_text_from_url("https://raw.githubusercontent.com/jdh-observer/JZx9gw7iwGxb/refs/heads/main/media/prompts/ocr_prompt.txt")

# Query GPT-4o
def query_gpt4(prompt, text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"{prompt}\n\n{text}"}]
    )
    return response.choices[0].message.content

gpt4_output = query_gpt4(ocr_prompt, ocr_output)

# Display comparison
run_comparison(
    ocr_output, human_output,
    "Human Corrected Transcript (Corrections in green)",
    "OCR Transcript (Errors in red)"
)

run_comparison(
    gpt4_output, human_output,
    "Human Corrected Transcript (Corrections in green)",
    "GPT-4 Corrected Transcript (Errors in red)"
)
```

<!-- #region jupyter={"outputs_hidden": false} -->
Here GPT-4o effectively corrects the OCR scan and abides by the prompt instructions, dropping the WER rate to a rate comparable to human transcriptions. GPT-4o also tends to omit German diacritics, abiding by the specific instructions in guiding the correction. This model’s strong performance on a high-quality image in a high-frequency language is another example of LLMs’s strengths for domains well-represented in its underlying training data. However, such optimal conditions are not always representative of the actual OCR scans digital historians need correcting. 
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"3sxqr": [{"id": "20666258/97BFFQS2", "source": "zotero"}], "gsecz": [{"id": "20666258/BG4SSJ9A", "source": "zotero"}]}} editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["hermeneutics"] -->
The use of few-shot prompting in the previous example demonstrates the importance of crafting effective instructions and contextual information to guide LLM outputs toward desired outcomes, a practice known as “prompt engineering.” Few-shot prompting represents just one approach among a rapidly growing number of prompt engineering techniques, each designed to leverage different aspects of LLMs' capacity for in-context learning. (<cite id="3sxqr"><a href="#zotero%7C20666258%2F97BFFQS2">(Vatsal &#38; Dubey, 2024)</a></cite>) While effective prompt engineering does not require extensive technical expertise, it demands clear communication skills and, crucially for historical applications, deep domain knowledge about the sources and contexts being analyzed. Such knowledge can help identify the hierarchies embedded in AI training data, allowing historians to craft prompts that mitigate predictable distortions while leveraging the models' strengths. As LLMs become increasingly integrated into digital workflows, prompt engineering may join skills like data visualization and network analysis as critical components of the historian's toolkit.


For those seeking to develop these capabilities, a useful starting point is DAIR.AI's Prompt Engineering Guide, which provides accessible examples of various prompting approaches and techniques. (<cite id="gsecz"><a href="#zotero%7C20666258%2FBG4SSJ9A">(Saravia, 2022/2022)</a></cite>)
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"b32ft": [{"id": "20666258/PD3VTHQ2", "source": "zotero"}], "hddxc": [{"id": "20666258/SC3QG7FE", "source": "zotero"}], "piwss": [{"id": "20666258/ASFKU59D", "source": "zotero"}]}} jupyter={"outputs_hidden": false} -->
Indeed, recent research reveals that LLMs become less effective when applied across diverse document types and languages. A study testing seven LLM models on historical Finnish texts found that prompt-based OCR correction proved "presently infeasible for this language," with even advanced models like GPT-4 achieving only modest improvements. The same models performed significantly better on English datasets, confirming the linguistic hierarchies observed in other applications. (<cite id="hddxc"><a href="#zotero%7C20666258%2FSC3QG7FE">(Kanerva et al., 2025)</a></cite>) Such patterns hold at larger scales. Swiss researchers tested fourteen different LLMs across diverse historical materials ranging from Byzantine papyri to twentieth-century newspapers. Their conclusion was stark: "Not only do [LLMs] not improve the original transcriptions, they usually degrade them." The degradation followed predictable patterns: worst performance on low-frequency texts like medieval Greek, and only modest improvements on contemporary materials in well-represented languages. Even GPT-4, the strongest performer, offered limited gains and frequently introduced hallucinations. As the researchers concluded, "LLM-based post-correction of historical transcripts [is] a rather distant prospect." (<cite id="b32ft"><a href="#zotero%7C20666258%2FPD3VTHQ2">(Boros et al., 2024)</a></cite>)

The challenges become even more apparent at industrial scales. Pleias, a French AI research group, created a massive dataset of LLM-corrected OCR scans: one billion words from French, English, German, and Italian cultural heritage repositories. Utilizing specialized models demonstrating real strengths in this domain, several systematic problems nonetheless surfaced in this process. Most notably, the models sometimes suffered from language-switching errors, where OCR mistakes trigger incorrect language detection and the models produce corrections in the wrong language entirely - such as turning English newspaper text into French, for example. Though specialized models demonstrated such errors less frequently than in general-purpose LLMs, such tendencies highlight the fundamental challenges of using AI for historical text correction. (<cite id="piwss"><a href="#zotero%7C20666258%2FASFKU59D">(Langlais, n.d.)</a></cite>)

Multimodal LLMs - models trained on both text and images - offer a potential mitigation to these limitations. Rather than relying solely on textual patterns, these models can incorporate visual information from the original document to further guide their corrections. Just as LLMs develop emergent abilities when trained across a wide array of texts, so too do multimodal LLMs develop similar capacities for “reading” images when trained on visual datasets paired with annotated text. The following example demonstrates how models trained in this manner can be prompted not just with textual instructions and the raw OCR scan, but with the underlying image itself. The image below also comes from the same German POW newspaper used earlier, but this example is chosen because it contains significant “noise” and distortion, rendering it more difficult for both OCR and generative AI models to parse.
<!-- #endregion -->

```python jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["figure-lotse-3-15-1945-*"]
from PIL import Image
from IPython.display import display, Image as IPImage
# Load and resize the image
ocr2_url = 'https://raw.githubusercontent.com/jdh-observer/JZx9gw7iwGxb/refs/heads/main/media/die_lotse_3-15-45_1.png'
image2 = Image.open('media/die_lotse_3-15-45_1.png')

# Set new dimensions for the resized image
new_width = 600
new_height = int(image2.height * (new_width / image2.width))

# Resize the image for better visualization
resized_image = image2.resize((new_width, new_height), Image.LANCZOS)

# Prepare the image for IPython display
resized_image.save("/tmp/resized_image.png") 
# Create formatted citation text with rich
citation_text = (
    "**Source:** *“Zum Geleit,”* *Die Lotse* (Camp McCain, Mississippi), 15 March 1945.\n"
    "In: Karl John Richard Arndt, editor. *German P.O.W. Camp Papers*. (Washington, D.C.: Library of Congress Photoduplication Service, 1965). Reel 9."
)
metadata = {
    "jdh": {
        "object": {
            "type": "image",
            "source": [
                citation_text
            ]
        }
    }
} 
display(IPImage(filename="/tmp/resized_image.png"), metadata=metadata) 
```

<!-- #region jupyter={"outputs_hidden": false} -->
The next example demonstrates this multimodal approach in action. We'll compare four approaches to the same distorted text: a human transcription (our baseline), the original OCR scan, GPT-4o correction using text-based prompting, and GPT-4o correction with both prompting and visual input. This approach reveals how providing the model with the original image can result in a more accurate correction.
<!-- #endregion -->

```python jupyter={"outputs_hidden": false}
# OCR correction with GPT-4o, few shot prompting and use of original image 

from openai import OpenAI
import base64
import requests

# Initialize OpenAI client (API key assumed to be loaded from env)
client = OpenAI()

# Helper: Encode image as base64
def encode_image_to_base64(image_url):
    response = requests.get(image_url)
    response.raise_for_status()
    return base64.b64encode(response.content).decode("utf-8")

# Text-only GPT-4o query (uses only ocr_prompt.txt)
def query_gpt4_text_only(prompt, ocr_output):
    full_prompt = f"{prompt}\n\nOCR Output:\n{ocr_output}\n\nPlease correct the OCR errors."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": full_prompt}],
        max_tokens=1500,
        temperature=0.0
    )
    return response.choices[0].message.content

# Multimodal GPT-4o query (uses few-shot + gpt_vision_prompt.txt)
def query_gpt4_with_image(prompt, ocr_output, base64_image):
    full_prompt = f"{prompt}\n\nOCR Output:\n{ocr_output}\n\nPlease correct the OCR errors."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    {"type": "text", "text": full_prompt}
                ]
            }
        ],
        max_tokens=1500,
        temperature=0.0
    )
    return response.choices[0].message.content

# --- File URLs ---
file_urls = {
    "image": "https://raw.githubusercontent.com/jdh-observer/JZx9gw7iwGxb/refs/heads/main/media/die_lotse_3-15-45_1.png",
    "ocr": "https://raw.githubusercontent.com/jdh-observer/JZx9gw7iwGxb/refs/heads/main/media/die_lotse_2_ocr_output.txt",
    "corrected": "https://raw.githubusercontent.com/jdh-observer/JZx9gw7iwGxb/refs/heads/main/media/die_lotse_2_human_correction.txt",
    "ocr_prompt": "https://raw.githubusercontent.com/jdh-observer/JZx9gw7iwGxb/refs/heads/main/media/prompts/ocr_prompt.txt",
    "gpt_vision_prompt": "https://raw.githubusercontent.com/jdh-observer/JZx9gw7iwGxb/refs/heads/main/media/prompts/gpt_vision_prompt.txt",
    "few_shot": "https://raw.githubusercontent.com/jdh-observer/JZx9gw7iwGxb/refs/heads/main/media/prompts/vision_few_shot.txt"
}

# --- Load content ---
ocr_output = fetch_text_from_url(file_urls["ocr"])
human_corrected_output = fetch_text_from_url(file_urls["corrected"])
base64_image = encode_image_to_base64(file_urls["image"])

# Prompts
text_only_prompt = fetch_text_from_url(file_urls["ocr_prompt"])
vision_prompt = fetch_text_from_url(file_urls["gpt_vision_prompt"])
few_shot_examples = fetch_text_from_url(file_urls["few_shot"])
multimodal_combined_prompt = f"{vision_prompt}\n\n{few_shot_examples}"

# --- GPT-4o Outputs ---
gpt4o_text_output = query_gpt4_text_only(text_only_prompt, ocr_output)
gpt4o_image_output = query_gpt4_with_image(multimodal_combined_prompt, ocr_output, base64_image)

# --- Comparisons ---
run_comparison(
    ocr_output,
    human_corrected_output,
    "Human Corrected Transcript (Corrections in green)",
    "OCR Transcript (Errors in red)"
)

run_comparison(
    gpt4o_text_output,
    human_corrected_output,
    "Human Corrected Transcript (Corrections in green)",
    "GPT-4o (Few Shot Prompting) Output (Errors in red)"
)

run_comparison(
    gpt4o_image_output,
    human_corrected_output,
    "Human Corrected Transcript (Corrections in green)",
    "GPT-4o (With Image + Few-Shot Prompt) Output (Errors in red)"
)

```

<!-- #region citation-manager={"citations": {"0zbfi": [{"id": "20666258/GTTJM2AS", "source": "zotero"}], "2jwjo": [{"id": "20666258/WTQSJCQI", "source": "zotero"}], "8lngk": [{"id": "20666258/5BFJP3BZ", "source": "zotero"}], "9c78k": [{"id": "20666258/GTTJM2AS", "source": "zotero"}]}} editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} -->
The results show improvement with each additional form of context provided to the model. Detailed instructions, examples, and a visual "ground truth" all help the model in better correcting the scan. However, how much improvement occurs depends on whether the source material resembles what the model learned during training.

Recent studies confirm multimodal AI's potential while revealing familiar limitations. Two research teams tested whether multimodal LLMs could outperform specialized handwriting text detection (HTD) software on historical manuscripts. The first study compared Google's Gemini against traditional handwriting detection models on historical manuscript collections, including Jeremy Bentham's English-language papers. While the HTD specialized models needed extensive training on specific examples to achieve highest accuracy, Gemini performed competitively on  English-language materials without any customization. (<cite id="9c78k"><a href="#zotero%7C20666258%2FGTTJM2AS">(W. Li et al., 2024)</a></cite>) A second study tested multimodal LLMs against HTD models created by Transkribus, using a custom dataset of English manuscripts from the 18th-19th centuries written in 33 different hands and of varying quality. The multimodal models consistently outperformed the specialized software, with Claude-Sonnet-3.5 achieving 20% better accuracy. Most importantly, these models proved effective at correcting errors from other transcription tools, reaching near-human levels of accuracy. (<cite id="8lngk"><a href="#zotero%7C20666258%2F5BFJP3BZ">(Humphries et al., 2024)</a></cite>) 

However, the same linguistic hierarchies emerge even in these models. Gemini's competitive performance was limited to English documents; when tested on German and French historical texts, it performed "dramatically worse," confirming that even visual training data reflects the same biases found in text-only models. (<cite id="0zbfi"><a href="#zotero%7C20666258%2FGTTJM2AS">(L. Li, 2024)</a></cite>) Rather than eliminating bias, multimodal models simply extend new forms of distortion into visual domains. Just as text-only LLMs tend to favor languages and time periods best represented in their training, multimodal models carry these same preferences into image processing.

These biases come from training on massive collections of internet images paired with text descriptions. The LAION-5B dataset, one of the few publicly available multimodal training collections, illustrates this problem clearly. Despite containing five billion image-text pairs in 100 languages, the dataset heavily favors English content. German represents only 6.6% of examples, while French accounts for 7.4% - disparities that likely explain why such models perform worse on non-English historical documents. As the dataset developers acknowledge, such internet-scraped collections risk "amplifying the social bias" of their sources. These datasets thus reproduce the same digital hierarchies that shaped text-only models, now mapped across visual terrains. (<cite id="2jwjo"><a href="#zotero%7C20666258%2FWTQSJCQI">(Schuhmann et al., 2022)</a></cite>)

What are the implications of these findings for digital historians considering LLMs for OCR correction? The current evidence indicates this domain is another contour in the jagged frontier of generative AI. LLMs have promise as a generalized OCR tool for aiding digitization of historical materials in high-frequency languages, using standard orthography, and created in recent decades enjoy the most effective correction by LLMs. However, these same models can degrade the accuracy of texts in underrepresented languages or linguistic norms. 

These findings suggest the need for strategic rather than wholesale adoption of LLM-based OCR correction. In domains where models excel, generative AI can be applied as a useful tool to circumvent research bottlenecks. Approaches such as few-shot prompting, model finetuning, and multimodal AI can help address limitations in those domains where models fall short. Yet such mitigations still present important limits. The efficiencies gained must be weighed against the risks of encoding systematic distortions into digitized historical sources. Understanding these trade-offs requires a new form of source criticism adapted to computational tools. Indeed, in employing generative AI historians must consider whose version of the past becomes computationally legible and whose remains obscured.
<!-- #endregion -->

<!-- #region editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} -->
## Mapping the Latent Past: How Historians Can Chart Generative AI’s Uneven Potential
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"2otyo": [{"id": "20666258/AMJV3VHE", "source": "zotero"}], "ajhvg": [{"id": "20666258/MCTRFU8M", "source": "zotero"}], "asr4q": [{"id": "20666258/EIKXH5UZ", "source": "zotero"}], "dlpro": [{"id": "20666258/IMAATWD8", "source": "zotero"}], "k22im": [{"id": "20666258/552SUT6F", "source": "zotero"}], "rh2wf": [{"id": "20666258/ESR92IRF", "source": "zotero"}], "ulw8r": [{"id": "20666258/9F2REN36", "source": "zotero"}]}} editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} -->
These case studies reveal how training data fundamentally shapes generative AI’s utility as a historical tool. By understanding this relationship, historians can better navigate an uneven landscape: well-mapped territories where these technologies excel, and the _terra incognita_ where they falter. The latent space of generative AI emerges from this particular cultural topography, one encoded by the linguistic, geographic, and temporal hierarchies of our digital past. In considering generative AI as a tool for data preparation and cleanup, scholars need legible maps illustrating this jagged topography. The same caution applies when considering generative AI as an interpretative method. 

The boundary between tool and method becomes visible in structured data extraction, where LLMs move from transcribing historical records to categorizing them. As Lauren Tilton reminds us, “[a]rtificial intelligence now tags, classifies, and filters the very sources historians encounter.” (<cite id="rh2wf"><a href="#zotero%7C20666258%2FESR92IRF">(Tilton, 2023)</a></cite>) A recent study at the Leibniz Institute of European History examines this practice in action. LLMs were used to identify relevant articles about the 1908 Messina earthquake from a a multilingual newspaper corpus, and offer a justification for their "decisions." In the German-language subset, most models retrieved relevant articles with better-than-80 percent accuracy, although results varied based on prompt approach and model size. When asked to defend their rationales, however, the models consistently relied on place names and dates in making their "judgements", and seldom based on the articles’ historical arguments. LLMs, in other words, could identify where and when with some confidence, but stumbled over why - a sharp contour between categorization and historical interpretation. (<cite id="dlpro"><a href="#zotero%7C20666258%2FIMAATWD8">(Oberbichler, 2025)</a></cite>)

If the Messina experiment mapped the limits of factual retrieval, a second study shows how those gulfs widen when LLMs attempt to reason about history itself. A team of historians recently pushed the ChatGPT model family into this interpretive terrain, posing 84 questions on the history of Fascist Italy to GPT-3.5, GPT-4, and GPT-4o. More than half of the answers contained factual errors, chronological confusion, or ambiguous interpretations. While GPT-4o outscored its predecessors, a deeper look exposed striking patterns: every model fixated on late-regime flashpoints (1938, 1940, 1943) while largely ignoring the movement’s formative years. The bias extended to citations: English-language scholarship was privileged over recent Italian literature even when the prompts were in Italian. Such clustering, the authors warn, does more than skew emphasis; it recycles dated narratives that “downplay aspects of the regime’s violent and totalitarian nature.” (<cite id="k22im"><a href="#zotero%7C20666258%2F552SUT6F">(De Ninno &#38; and Lacriola, 2025)</a></cite>) Such concerning tendencies reveal how LLMs can perpetuate historiographical frames that scholars have spent decades correcting. 

One approach to respond to the limitations of LLMs is “agentic” AI systems that empower models to seek new data beyond their frozen training sets. By connecting an LLM to search engines, domain databases, code interpreters, or retrieval-augmented-generation (RAG) pipelines, designers hope to overcome hallucination and fill gaps in a model’s training. Harvard’s Library Innovation Lab, for instance, has released WARC-GPT, a RAG system that empowers LLMs to query web-archive collections to aid research requests. (<cite id="ajhvg"><a href="#zotero%7C20666258%2FMCTRFU8M">(Cargnelutti et al., 2024)</a></cite>) STORM, from Stanford’s Open Virtual Assistant Lab, goes further: this system uses a panel of specialist LLM “experts” to scour the web and assemble Wikipedia-style articles on demand. (<cite id="asr4q"><a href="#zotero%7C20666258%2FEIKXH5UZ">(Shao et al., 2024)</a></cite>) Yet, as Benjamin Schmidt cautions, these systems “not only give us new haystacks to search in; they also change the types of needles people will find.” (<cite id="ulw8r"><a href="#zotero%7C20666258%2F9F2REN36">(Schmidt, 2023)</a></cite>) If the relationship holds between training data and performance, then agentic AI merely shifts the modalities of bias, rather than abolishing it. Early RAG studies appear to bear this out. (<cite id="2otyo"><a href="#zotero%7C20666258%2FAMJV3VHE">(Hu et al., 2024)</a></cite>)  The persistence of bias across these sophisticated systems confirms that the hierarchies mapped in our case studies are not technical bugs to be patched but fundamental features of how AI systems inherit and amplify the inequalities of their training data.

The project of mapping the latent past - understanding how AI systems encode and reshape historical memory - extends beyond identifying current limitations. A growing body of research has established the uneven potential of generative AI as a historical tool. However, this technology’s rapid development and potent social impact means this terrain can shift in unexpected ways. Historians should create disciplinary atlases that guide future exploration over this protean space. Further benchmarks are needed to probe historical fluency where today’s models are silent: in less-represented languages, non-Western chronologies, and marginalized perspectives. Identifying these silences can inform the development of specialized corpora that can be used to make muted voices audible both to the models and to the scholars who use them. Indeed, as LLMs become increasingly embedded in our digital lives, historians should continue to use their expertise to inform the larger debates about the ethical, political, and environmental costs of generative AI.  By approaching LLMs critically, ethically, and collaboratively, digital historians are contributing to Roy Rosenzweig’s vision of examining the “unheard-of historical abundance” of the digital age with tools that deepen, rather than distort, our understanding of the past.
<!-- #endregion -->

<!-- #region editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} -->
## Acknowledgements
<!-- #endregion -->

<!-- #region editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} -->
I am grateful to Abraham Gibson for extending an invitation to present the preliminary research findings of this article with the Digital History Working Group in May 2022, organized by the Consortium For History of Science, Technology, and Medicine. I would also like to express my appreciation to my colleagues William Mattingly, Patrick Wadden, and Ian Crowe for their insightful commentary on the article, and to the editorial staff and reviewers for the Journal of Digital History. This article was facilitated by a sabbatical semester generously granted by the Office of Academic Affairs at Belmont Abbey College.
<!-- #endregion -->

<!-- #region editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} -->
## Bibliography
<!-- #endregion -->

<!-- #region editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} -->
<!-- BIBLIOGRAPHY START -->
<div class="csl-bib-body">
  <div class="csl-entry"><i id="zotero|20666258/ZKMMJTTK"></i>Barton, N. T. L., Paul Resnick, and Genie. (2019, May 22). Algorithmic bias detection and mitigation: Best practices and policies to reduce consumer harms. <i>Brookings</i>. <a href="https://www.brookings.edu/research/algorithmic-bias-detection-and-mitigation-best-practices-and-policies-to-reduce-consumer-harms/">https://www.brookings.edu/research/algorithmic-bias-detection-and-mitigation-best-practices-and-policies-to-reduce-consumer-harms/</a></div>
  <div class="csl-entry"><i id="zotero|20666258/MAEXPBX2"></i>Bender, E., Gebru, T., McMillan-Major, A., &#38; Mitchell, M. (n.d.). <i>On the Dangers of Stochastic Parrots | Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency</i>. Retrieved March 27, 2023, from <a href="https://dl.acm.org/doi/10.1145/3442188.3445922">https://dl.acm.org/doi/10.1145/3442188.3445922</a></div>
  <div class="csl-entry"><i id="zotero|20666258/P4D7WWG3"></i>Benjamin, R. (2019). <i>Race After Technology: Abolitionist Tools for the New Jim Code</i> (1st edition). Polity.</div>
  <div class="csl-entry"><i id="zotero|20666258/ZW8DJ3K3"></i>Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O’Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., Skowron, A., Sutawika, L., &#38; Wal, O. van der. (2023). <i>Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling</i> (arXiv:2304.01373). arXiv. <a href="https://doi.org/10.48550/arXiv.2304.01373">https://doi.org/10.48550/arXiv.2304.01373</a></div>
  <div class="csl-entry"><i id="zotero|20666258/PD3VTHQ2"></i>Boros, E., Ehrmann, M., Romanello, M., Najem-Meyer, S., &#38; Kaplan, F. (2024). Post-Correction of Historical Text Transcripts with Large Language Models: An Exploratory Study. In Y. Bizzoni, S. Degaetano-Ortlieb, A. Kazantseva, &#38; S. Szpakowicz (Eds.), <i>Proceedings of the 8th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature (LaTeCH-CLfL 2024)</i> (pp. 133–159). Association for Computational Linguistics. <a href="https://aclanthology.org/2024.latechclfl-1.14/">https://aclanthology.org/2024.latechclfl-1.14/</a></div>
  <div class="csl-entry"><i id="zotero|20666258/WHCGSCI5"></i>Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., … Amodei, D. (2020). <i>Language Models are Few-Shot Learners</i> (arXiv:2005.14165). arXiv. <a href="http://arxiv.org/abs/2005.14165">http://arxiv.org/abs/2005.14165</a></div>
  <div class="csl-entry"><i id="zotero|20666258/MCTRFU8M"></i>Cargnelutti, M., Mukk, K., &#38; Stanton, C. (2024, February 12). <i>WARC-GPT: An Open-Source Tool for Exploring Web Archives Using AI | Library Innovation Lab</i>. Library Innovation Lab Blog, Harvard Law Library. <a href="https://lil.law.harvard.edu/blog/2024/02/12/warc-gpt-an-open-source-tool-for-exploring-web-archives-with-ai/">https://lil.law.harvard.edu/blog/2024/02/12/warc-gpt-an-open-source-tool-for-exploring-web-archives-with-ai/</a></div>
  <div class="csl-entry"><i id="zotero|20666258/A6DI9F7T"></i>Chartier, M., Dakkoune, N., Bourgeois, G., &#38; Jean, S. (2025). HiBenchLLM: Historical Inquiry Benchmarking for Large Language Models. <i>Data &#38; Knowledge Engineering</i>, <i>156</i>, 102383. <a href="https://doi.org/10.1016/j.datak.2024.102383">https://doi.org/10.1016/j.datak.2024.102383</a></div>
  <div class="csl-entry"><i id="zotero|20666258/GETAJ6CA"></i>Christiano, P., Leike, J., Brown, T. B., Martic, M., Legg, S., &#38; Amodei, D. (2023). <i>Deep reinforcement learning from human preferences</i> (arXiv:1706.03741). arXiv. <a href="https://doi.org/10.48550/arXiv.1706.03741">https://doi.org/10.48550/arXiv.1706.03741</a></div>
  <div class="csl-entry"><i id="zotero|20666258/2GJME5SQ"></i>Clavert, F. (2024, June 12). Creativy and AI - recording. <i>C2DH EN</i>. <a href="https://www.uni.lu/c2dh-en/articles/creativy-and-ai-recording/">https://www.uni.lu/c2dh-en/articles/creativy-and-ai-recording/</a></div>
  <div class="csl-entry"><i id="zotero|20666258/FCDIMVCZ"></i>College Board. (n.d.). <i>Program Summary Report (2024)</i>. <a href="https://apcentral.collegeboard.org/media/pdf/program-summary-report-2024.pdf">https://apcentral.collegeboard.org/media/pdf/program-summary-report-2024.pdf</a></div>
  <div class="csl-entry"><i id="zotero|20666258/W569JM2K"></i>Crawford, K. (2021). <i>Atlas of AI: Power, Politics, and the Planetary Costs of Artificial Intelligence</i>. Yale University Press.</div>
  <div class="csl-entry"><i id="zotero|20666258/89T9PJV9"></i>Crawford, K., &#38; Paglen, T. (n.d.). <i>Excavating AI</i>. Excavating AI. Retrieved March 28, 2023, from <a href="https://excavating.ai">https://excavating.ai</a></div>
  <div class="csl-entry"><i id="zotero|20666258/BIZ89DQ7"></i>Danilova, V., &#38; Aangenendt, G. (2025). Post-OCR Correction of Historical German Periodicals using LLMs. In Š. A. Holdt, N. Ilinykh, B. Scalvini, M. Bruton, I. N. Debess, &#38; C. M. Tudor (Eds.), <i>Proceedings of the Third Workshop on Resources and Representations for Under-Resourced Languages and Domains (RESOURCEFUL-2025)</i> (pp. 120–129). University of Tartu Library, Estonia. <a href="https://aclanthology.org/2025.resourceful-1.26/">https://aclanthology.org/2025.resourceful-1.26/</a></div>
  <div class="csl-entry"><i id="zotero|20666258/ZSIP6FKE"></i>Dasu, T., &#38; Johnson, T. (2003). <i>Exploratory Data Mining and Data Cleaning</i> (1st edition). Wiley-Interscience.</div>
  <div class="csl-entry"><i id="zotero|20666258/552SUT6F"></i>De Ninno, F., &#38; and Lacriola, M. (2025). Mussolini and ChatGPT. Examining the Risks of A.I. writing Historical Narratives on Fascism. <i>Journal of Modern Italian Studies</i>, <i>30</i>(2), 187–209. <a href="https://doi.org/10.1080/1354571X.2025.2457250">https://doi.org/10.1080/1354571X.2025.2457250</a></div>
  <div class="csl-entry"><i id="zotero|20666258/A427KHHQ"></i>Dong, Q., Li, L., Dai, D., Zheng, C., Ma, J., Li, R., Xia, H., Xu, J., Wu, Z., Liu, T., Chang, B., Sun, X., Li, L., &#38; Sui, Z. (2024). <i>A Survey on In-context Learning</i> (arXiv:2301.00234). arXiv. <a href="https://doi.org/10.48550/arXiv.2301.00234">https://doi.org/10.48550/arXiv.2301.00234</a></div>
  <div class="csl-entry"><i id="zotero|20666258/2KU6Q2RE"></i>Franklin, J. H. (1990, July 27). <i>Oral History Interview with John Hope Franklin,  Interview A-0339.. Conducted by John Egerton. Southern Oral History Program Collection (#4007).</i> (4007) [Interview]. Southern Oral History Program Collection. <a href="https://docsouth.unc.edu/sohp/A-0339/menu.html">https://docsouth.unc.edu/sohp/A-0339/menu.html</a></div>
  <div class="csl-entry"><i id="zotero|20666258/BERD6ARS"></i>Gebru, T. (2020). Race and Gender. In M. D. Dubber, F. Pasquale, &#38; S. Das (Eds.), <i>The Oxford Handbook of Ethics of AI</i> (p. 0). Oxford University Press. <a href="https://doi.org/10.1093/oxfordhb/9780190067397.013.16">https://doi.org/10.1093/oxfordhb/9780190067397.013.16</a></div>
  <div class="csl-entry"><i id="zotero|20666258/ENVC5ZIJ"></i>Gehman, S., Gururangan, S., Sap, M., Choi, Y., &#38; Smith, N. A. (2020). <i>RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models</i> (arXiv:2009.11462). arXiv. <a href="http://arxiv.org/abs/2009.11462">http://arxiv.org/abs/2009.11462</a></div>
  <div class="csl-entry"><i id="zotero|20666258/7M6MP3NI"></i>Graham, S., Milligan, I., &#38; Weingart, S. (2015). <i>Exploring Big Historical Data: The Historian’s Macroscope</i> (Reprint edition). Icp.</div>
  <div class="csl-entry"><i id="zotero|20666258/E38ZE6TS"></i>Hauser, J., Kondor, D., Reddish, J., Benam, M., Cioni, E., Villa, F., Bennett, J. S., Hoyer, D., Francois, P., Turchin, P., &#38; Rio-Chanona, R. M. del. (2024, November 13). <i>Large Language Models’ Expert-level Global History Knowledge Benchmark (HiST-LLM)</i>. The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track. <a href="https://openreview.net/forum?id=xlKeMuyoZ5#discussion">https://openreview.net/forum?id=xlKeMuyoZ5#discussion</a></div>
  <div class="csl-entry"><i id="zotero|20666258/6HHA7DFR"></i>Hendrycks, D. (2023). <i>Measuring Massive Multitask Language Understanding</i>. <a href="https://github.com/hendrycks/test">https://github.com/hendrycks/test</a> (Original work published 2020)</div>
  <div class="csl-entry"><i id="zotero|20666258/Q269X8CB"></i>Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., &#38; Steinhardt, J. (2021). <i>Measuring Massive Multitask Language Understanding</i> (arXiv:2009.03300). arXiv. <a href="http://arxiv.org/abs/2009.03300">http://arxiv.org/abs/2009.03300</a></div>
  <div class="csl-entry"><i id="zotero|20666258/AMJV3VHE"></i>Hu, M., Wu, H., Guan, Z., Zhu, R., Guo, D., Qi, D., &#38; Li, S. (2024). <i>No Free Lunch: Retrieval-Augmented Generation Undermines Fairness in LLMs, Even for Vigilant Users</i>. <a href="https://openreview.net/forum?id=cphaRg46jD">https://openreview.net/forum?id=cphaRg46jD</a></div>
  <div class="csl-entry"><i id="zotero|20666258/5BFJP3BZ"></i>Humphries, M., Leddy, L. C., Downton, Q., Legace, M., McConnell, J., Murray, I., &#38; Spence, E. (2024). <i>Unlocking the Archives: Large Language Models Achieve State-of-the-Art Performance on the Transcription of Handwritten Historical Documents</i> (SSRN Scholarly Paper No. 5006071). Social Science Research Network. <a href="https://doi.org/10.2139/ssrn.5006071">https://doi.org/10.2139/ssrn.5006071</a></div>
  <div class="csl-entry"><i id="zotero|20666258/ZHJK8JPH"></i>Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y., Dai, W., Madotto, A., &#38; Fung, P. (2023). Survey of Hallucination in Natural Language Generation. <i>ACM Computing Surveys</i>, <i>55</i>(12), 1–38. <a href="https://doi.org/10.1145/3571730">https://doi.org/10.1145/3571730</a></div>
  <div class="csl-entry"><i id="zotero|20666258/SC3QG7FE"></i>Kanerva, J., Ledins, C., Käpyaho, S., &#38; Ginter, F. (2025). <i>OCR Error Post-Correction with LLMs in Historical Documents: No Free Lunches</i> (arXiv:2502.01205). arXiv. <a href="https://doi.org/10.48550/arXiv.2502.01205">https://doi.org/10.48550/arXiv.2502.01205</a></div>
  <div class="csl-entry"><i id="zotero|20666258/RRUNZDAM"></i>Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., &#38; Amodei, D. (2020). <i>Scaling Laws for Neural Language Models</i> (arXiv:2001.08361). arXiv. <a href="https://doi.org/10.48550/arXiv.2001.08361">https://doi.org/10.48550/arXiv.2001.08361</a></div>
  <div class="csl-entry"><i id="zotero|20666258/PVPPHZ56"></i>Katz, D. M. (2022). <i>GPT Takes the Bar Exam</i> (arXiv:2212.14402). arXiv. <a href="https://doi.org/10.48550/arXiv.2212.14402">https://doi.org/10.48550/arXiv.2212.14402</a></div>
  <div class="csl-entry"><i id="zotero|20666258/9N7V3WDP"></i>Koenecke, A., Choi, A. S. G., Mei, K. X., Schellmann, H., &#38; Sloane, M. (2024). <i>Careless Whisper: Speech-to-Text Hallucination Harms</i> (arXiv:2402.08021). arXiv. <a href="https://doi.org/10.48550/arXiv.2402.08021">https://doi.org/10.48550/arXiv.2402.08021</a></div>
  <div class="csl-entry"><i id="zotero|20666258/ASFKU59D"></i>Langlais, P.-C. (n.d.). <i>Post-OCR-Correction: 1 billion words dataset of automated OCR correction by LLM</i>. Retrieved October 14, 2024, from <a href="https://huggingface.co/blog/Pclanglais/post-ocr-correction">https://huggingface.co/blog/Pclanglais/post-ocr-correction</a></div>
  <div class="csl-entry"><i id="zotero|20666258/DM9N78FC"></i>Lehečka, J., Švec, J., Psutka, J. V., &#38; Ircing, P. (2023). Transformer-based Speech Recognition Models for Oral History Archives in English, German, and Czech. <i>INTERSPEECH 2023</i>, 201–205. <a href="https://doi.org/10.21437/Interspeech.2023-872">https://doi.org/10.21437/Interspeech.2023-872</a></div>
  <div class="csl-entry"><i id="zotero|20666258/GTTJM2AS"></i>Li, L. (2024). <i>Handwriting Recognition in Historical Documents with Multimodal LLM</i> (arXiv:2410.24034). arXiv. <a href="https://doi.org/10.48550/arXiv.2410.24034">https://doi.org/10.48550/arXiv.2410.24034</a></div>
  <div class="csl-entry"><i id="zotero|20666258/FWUHDSFT"></i>Li, W., Ma, R., Wu, J., Gu, C., Peng, J., Len, J., Zhang, S., Yan, H., Lin, D., &#38; He, C. (2024). <i>FoundaBench: Evaluating Chinese Fundamental Knowledge Capabilities of Large Language Models</i> (arXiv:2404.18359). arXiv. <a href="https://doi.org/10.48550/arXiv.2404.18359">https://doi.org/10.48550/arXiv.2404.18359</a></div>
  <div class="csl-entry"><i id="zotero|20666258/35H7ZC5Z"></i>Mai, Y., &#38; Liang, P. (2024, May 1). <i>Massive Multitask Language Understanding (MMLU) on HELM</i> [Blog]. Center for Research on Foundation Models, Stanford University. <a href="https://crfm.stanford.edu/2024/05/01/helm-mmlu.html">https://crfm.stanford.edu/2024/05/01/helm-mmlu.html</a></div>
  <div class="csl-entry"><i id="zotero|20666258/8ISS2NP3"></i>Marshall, L. (2020, October 20). <i>The Strange World of AP U.S. History</i>. CONTINGENT. <a href="https://contingentmagazine.org/2020/10/20/apush/">https://contingentmagazine.org/2020/10/20/apush/</a></div>
  <div class="csl-entry"><i id="zotero|20666258/9QVGGFHF"></i>Milligan, I. (2013). Illusionary Order: Online Databases, Optical Character Recognition, and Canadian History, 1997–2010. <i>Canadian Historical Review</i>, <i>94</i>(4), 540–569. <a href="https://doi.org/10.3138/chr.694">https://doi.org/10.3138/chr.694</a></div>
  <div class="csl-entry"><i id="zotero|20666258/7AQMGH6M"></i>Mollick, E. (2024). <i>Co-intelligence: living and working with AI</i> (Unabridged). Books on Tape. <a href="https://www.overdrive.com/search?q=B866FB9A-0956-441A-BA5A-7BC03F491FA3">https://www.overdrive.com/search?q=B866FB9A-0956-441A-BA5A-7BC03F491FA3</a></div>
  <div class="csl-entry"><i id="zotero|20666258/V49364MS"></i>Muehlberger, G., Seaward, L., Terras, M., Ares Oliveira, S., Bosch, V., Bryan, M., Colutto, S., Déjean, H., Diem, M., Fiel, S., Gatos, B., Greinoecker, A., Grüning, T., Hackl, G., Haukkovaara, V., Heyer, G., Hirvonen, L., Hodel, T., Jokinen, M., … Zagoris, K. (2019). Transforming scholarship in the archives through handwritten text recognition: Transkribus as a case study. <i>Journal of Documentation</i>, <i>75</i>(5), 954–976. <a href="https://doi.org/10.1108/JD-07-2018-0114">https://doi.org/10.1108/JD-07-2018-0114</a></div>
  <div class="csl-entry"><i id="zotero|20666258/JIJPRUQN"></i>Ninh, B. (2005, March 17). <i>Interview with Bao Ninh, OH0435. Conducted by Richard B. Verrone and Khanh Le.</i> [Interview]. Vietnam Center and Sam Johnson Vietnam Archive, Texas Tech University,. <a href="https://www.vietnam.ttu.edu/virtualarchive/items.php?item=OH0435">https://www.vietnam.ttu.edu/virtualarchive/items.php?item=OH0435</a></div>
  <div class="csl-entry"><i id="zotero|20666258/84CCZGZA"></i>Noble, S. U. (2018). <i>Algorithms of Oppression: How Search Engines Reinforce Racism</i> (Illustrated edition). NYU Press.</div>
  <div class="csl-entry"><i id="zotero|20666258/BRJE5S95"></i>Nori, H., King, N., McKinney, S. M., Carignan, D., &#38; Horvitz, E. (2023). <i>Capabilities of GPT-4 on Medical Challenge Problems</i> (arXiv:2303.13375). arXiv. <a href="http://arxiv.org/abs/2303.13375">http://arxiv.org/abs/2303.13375</a></div>
  <div class="csl-entry"><i id="zotero|20666258/IMAATWD8"></i>Oberbichler, S. (2025, January 31). LLM Biases: Expected and Unexpected Model Design Effects in Historical Newspaper Article Extraction on the Messina Earthquake [Billet]. <i>DH Lab</i>. <a href="https://doi.org/10.58079/137qr">https://doi.org/10.58079/137qr</a></div>
  <div class="csl-entry"><i id="zotero|20666258/IXVBRSGM"></i>Oberbichler, S., &#38; Petz, C. (2025). <i>Working Paper: Implementing Generative AI in the Historical Studies</i>. Zenodo. <a href="https://doi.org/10.5281/zenodo.14924737">https://doi.org/10.5281/zenodo.14924737</a></div>
  <div class="csl-entry"><i id="zotero|20666258/7MEV3F4T"></i>O’brien, M. (2023, August 1). <i>Chatbots sometimes make things up. Is AI’s hallucination problem fixable?</i> AP News. <a href="https://apnews.com/article/artificial-intelligence-hallucination-chatbots-chatgpt-falsehoods-ac4672c5b06e6f91050aa46ee731bcf4">https://apnews.com/article/artificial-intelligence-hallucination-chatbots-chatgpt-falsehoods-ac4672c5b06e6f91050aa46ee731bcf4</a></div>
  <div class="csl-entry"><i id="zotero|20666258/IDG87H2E"></i>OpenAI. (2023). <i>GPT-4 Technical Report</i> (arXiv:2303.08774). arXiv. <a href="http://arxiv.org/abs/2303.08774">http://arxiv.org/abs/2303.08774</a></div>
  <div class="csl-entry"><i id="zotero|20666258/5UK7S2IS"></i>Peng, P., Yan, B., Watanabe, S., &#38; Harwath, D. (2023). <i>Prompting the Hidden Talent of Web-Scale Speech Models for Zero-Shot Task Generalization</i> (arXiv:2305.11095). arXiv. <a href="https://doi.org/10.48550/arXiv.2305.11095">https://doi.org/10.48550/arXiv.2305.11095</a></div>
  <div class="csl-entry"><i id="zotero|20666258/C5683JU6"></i>Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., &#38; Sutskever, I. (2021). <i>Learning Transferable Visual Models From Natural Language Supervision</i> (arXiv:2103.00020). arXiv. <a href="http://arxiv.org/abs/2103.00020">http://arxiv.org/abs/2103.00020</a></div>
  <div class="csl-entry"><i id="zotero|20666258/66HNW5BJ"></i>Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., &#38; Sutskever, I. (2022). <i>Robust Speech Recognition via Large-Scale Weak Supervision</i> (arXiv:2212.04356). arXiv. <a href="http://arxiv.org/abs/2212.04356">http://arxiv.org/abs/2212.04356</a></div>
  <div class="csl-entry"><i id="zotero|20666258/M64T96PV"></i>Ritchie, D. A. (2003). <i>Doing Oral History: a Practical Guide</i> (2nd ed). Oxford University Press, USA.</div>
  <div class="csl-entry"><i id="zotero|20666258/2NNI9P9W"></i>Rochester Institute of Technology. (n.d.). <i>Artificial intelligence aids cultural heritage researchers documenting and teaching oral histories</i>. Artificial Intelligence Aids Cultural Heritage Researchers Documenting and Teaching Oral Histories. Retrieved October 10, 2024, from <a href="https://www.rit.edu/news/artificial-intelligence-aids-cultural-heritage-researchers-documenting-and-teaching-oral">https://www.rit.edu/news/artificial-intelligence-aids-cultural-heritage-researchers-documenting-and-teaching-oral</a></div>
  <div class="csl-entry"><i id="zotero|20666258/37INR4W2"></i>Rosenzweig, R. (2003). Scarcity or Abundance? Preserving the Past in a Digital Era. <i>The American Historical Review</i>, <i>108</i>(3), 735–762. <a href="https://doi.org/10.1086/ahr/108.3.735">https://doi.org/10.1086/ahr/108.3.735</a></div>
  <div class="csl-entry"><i id="zotero|20666258/BG4SSJ9A"></i>Saravia, E. (2022). <i>Prompt Engineering Guide</i>. <a href="https://github.com/dair-ai/Prompt-Engineering-Guide">https://github.com/dair-ai/Prompt-Engineering-Guide</a> (Original work published 2022)</div>
  <div class="csl-entry"><i id="zotero|20666258/BQPMNA6D"></i>Sawmya, S., Adler, M., &#38; Shavit, N. (2025). <i>The Birth of Knowledge: Emergent Features across Time, Space, and Scale in Large Language Models</i> (arXiv:2505.19440). arXiv. <a href="https://doi.org/10.48550/arXiv.2505.19440">https://doi.org/10.48550/arXiv.2505.19440</a></div>
  <div class="csl-entry"><i id="zotero|20666258/9F2REN36"></i>Schmidt, B. (2023). Representation Learning. <i>The American Historical Review</i>, <i>128</i>(3), 1350–1353. <a href="https://doi.org/10.1093/ahr/rhad363">https://doi.org/10.1093/ahr/rhad363</a></div>
  <div class="csl-entry"><i id="zotero|20666258/WTQSJCQI"></i>Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., Schramowski, P., Kundurthy, S., Crowson, K., Schmidt, L., Kaczmarczyk, R., &#38; Jitsev, J. (2022). <i>LAION-5B: An open large-scale dataset for training next generation image-text models</i> (arXiv:2210.08402). arXiv. <a href="https://doi.org/10.48550/arXiv.2210.08402">https://doi.org/10.48550/arXiv.2210.08402</a></div>
  <div class="csl-entry"><i id="zotero|20666258/K7JFQAA5"></i>Schultz, E. (2024, February 12). <i>[Tutorial] Using Whisper to Transcribe Oral Interviews – CSS @ IPP</i>. <a href="https://www.css.cnrs.fr/using-whisper-to-transcribe-oral-interviews/">https://www.css.cnrs.fr/using-whisper-to-transcribe-oral-interviews/</a></div>
  <div class="csl-entry"><i id="zotero|20666258/EIKXH5UZ"></i>Shao, Y., Jiang, Y., Kanell, T. A., Xu, P., Khattab, O., &#38; Lam, M. S. (2024). <i>Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models</i> (arXiv:2402.14207). arXiv. <a href="https://doi.org/10.48550/arXiv.2402.14207">https://doi.org/10.48550/arXiv.2402.14207</a></div>
  <div class="csl-entry"><i id="zotero|20666258/Z9B75488"></i>Strickland, E. (2021, February 1). <i>OpenAI’s GPT-3 Speaks! (Kindly Disregard Toxic Language) - IEEE Spectrum</i>. OpenAI’s GPT-3 Speaks! (Kindly Disregard Toxic Language) - IEEE Spectrum. <a href="https://spectrum.ieee.org/open-ais-powerful-text-generating-tool-is-ready-for-business">https://spectrum.ieee.org/open-ais-powerful-text-generating-tool-is-ready-for-business</a></div>
  <div class="csl-entry"><i id="zotero|20666258/ESR92IRF"></i>Tilton, L. (2023). Relating to Historical Sources. <i>The American Historical Review</i>, <i>128</i>(3), 1354–1359. <a href="https://doi.org/10.1093/ahr/rhad365">https://doi.org/10.1093/ahr/rhad365</a></div>
  <div class="csl-entry"><i id="zotero|20666258/9IDUQQET"></i>Underwood, T. (2021). <i>Mapping the Latent Spaces of Culture</i>. <a href="https://hcommons.org/deposits/item/hc:41973/">https://hcommons.org/deposits/item/hc:41973/</a></div>
  <div class="csl-entry"><i id="zotero|20666258/P96ZKU8N"></i>Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., &#38; Polosukhin, I. (2023). <i>Attention Is All You Need</i> (arXiv:1706.03762). arXiv. <a href="http://arxiv.org/abs/1706.03762">http://arxiv.org/abs/1706.03762</a></div>
  <div class="csl-entry"><i id="zotero|20666258/97BFFQS2"></i>Vatsal, S., &#38; Dubey, H. (2024). <i>A Survey of Prompt Engineering Methods in Large Language Models for Different NLP Tasks</i> (arXiv:2407.12994). arXiv. <a href="https://doi.org/10.48550/arXiv.2407.12994">https://doi.org/10.48550/arXiv.2407.12994</a></div>
  <div class="csl-entry"><i id="zotero|20666258/7ERZCN5G"></i>Wong, A. (2018, June 13). The Controversy Over Just How Much History AP World History Should Cover. <i>The Atlantic</i>. <a href="https://www.theatlantic.com/education/archive/2018/06/ap-world-history-controversy/562778/">https://www.theatlantic.com/education/archive/2018/06/ap-world-history-controversy/562778/</a></div>
  <div class="csl-entry"><i id="zotero|20666258/W4XUTNCR"></i>Xu, R., Wang, Z., Fan, R.-Z., &#38; Liu, P. (2024). <i>Benchmarking Benchmark Leakage in Large Language Models</i> (arXiv:2404.18824). arXiv. <a href="https://doi.org/10.48550/arXiv.2404.18824">https://doi.org/10.48550/arXiv.2404.18824</a></div>
</div>
<!-- BIBLIOGRAPHY END -->
<!-- #endregion -->

```python editable=true jupyter={"outputs_hidden": false} slideshow={"slide_type": ""}

```
