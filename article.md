---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["title"] -->
# Prompting the Past: Large Language Models as Versatile Tools for Digital Historians

<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} tags=["contributor"] -->
 ### anonym
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["copyright"] -->
[![cc-by](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/) 
©<AUTHOR or ORGANIZATION / FUNDER>. Published by De Gruyter in cooperation with the University of Luxembourg Centre for Contemporary and Digital History. This is an Open Access article distributed under the terms of the [Creative Commons Attribution License CC-BY](https://creativecommons.org/licenses/by/4.0/)

<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["keywords"] -->
Large language models, artifical intelligence, machine learning, historical methodology, optical character recognition, oral history, prompt engineering
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["abstract"] -->
This article examines how digital historians are using large language models (LLMs) in their research and teaching, along with the critical and ethical debates surrounding their use. The article first assesses the historical capacities of LLMs as measured by machine learning benchmarks, and how such assessments can help historians understand the capacities and limits of these technologies. The utility of LLMs as digital tools are then demonstrated through a series of case studies using GPT-4 and other generative AI models for oral history transcriptions, correcting optical character recognition (OCR) errors, and metadata extraction. These case studies also demonstrate how frameworks for using LLMs, such as prompt engineering and retrieval augmented generation (RAG), are used to ground LLM outputs for consistency and greater accuracy. Acknowledging the significant ethical challenges posed by LLMs, the article emphasizes the need for critical engagement and the development of responsible frameworks for implementing these technologies in historical scholarship. By combining disciplinary expertise with innovative computational approaches, historians are discovering new ways to navigate the "unheard-of historical abundance" of the digital age, contributing to approaches to generative AI that enriches, rather than distorts, our understanding of the past.


<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} -->
## Introduction
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"5w5sr": [{"id": "27937/XQYUJV5F", "source": "zotero"}], "fgell": [{"id": "27937/CJYNFHVI", "source": "zotero"}], "hiex8": [{"id": "27937/GHGWH4HI", "source": "zotero"}], "uo7pa": [{"id": "27937/L2ILKERU", "source": "zotero"}]}} jupyter={"outputs_hidden": false} -->
In 2003, Roy Rosenzweig predicted that digital historians would need to develop new techniques "to research, write, and teach in a world of unheard-of historical abundance." (<cite id="fgell"><a href="#zotero%7C27937%2FCJYNFHVI">Rosenzweig, “Scarcity or Abundance?”</a></cite>) Over the past two decades historians have risen to this challenge, embracing digital mapping, network analysis, distant reading of large text collections, and machine learning as part of their growing methodological toolkit. (<cite id="uo7pa"><a href="#zotero%7C27937%2FL2ILKERU">Graham, Milligan, and Weingart, <i>Exploring Big Historical Data</i>.</a></cite>) Generative artificial intelligence (AI) has emerged as another potential tool for historians, particularly large language models (LLMs), the most prominent form of this technology. These models possess striking capacities to generate, interpret, and manipulate data across a range of modalities. The rapidly-expanding scope of these capabilities and their limits remain intensely debated, as do their broader social, economic, cultural, and environmental impacts. Yet while still an emerging technology, historians are already demonstrating generative AI's potential as a versatile digital tool. Historians are also contributing to the critical discourse surrounding this new domain, raising key questions about how these models are created, their propensity to reinforce existing inequalities, and their potential to distort our understanding of the past. (<cite id="5w5sr"><a href="#zotero%7C27937%2FXQYUJV5F">Meadows and Sternfeld, “Artificial Intelligence and the Practice of History.”</a></cite>)

This article contributes to these debates by demonstrating how digital historians are using generative AI to explore the past and the disciplinary contributions historians can offer in these broader debates concerning this technology. (<cite id="hiex8"><a href="#zotero%7C27937%2FGHGWH4HI">Dzieza, “What AI Can Do for Historians.”</a></cite>) We begin by assessing the metrics commonly used to measure the historical knowledge of LLMs, and examine how such metrics can give us insights into the capacities and limits this technology. We then examine how generative AI can be used in tasks as varied as preparing datasets, exploring text collections, and offering novel (and controversial) methods of representing the past. We conclude with a call to historians to contribute to ongoing research and debates concerning the ethical use of generative AI. Given the rapid pace of innovation in this field, it is crucial that the profession addresses the implications of this technology for our research and teaching. Historians will have much to offer in contextualizing these technologies and their potential impacts on society.
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} -->
## What Do AIs Know About History? Assessing LLMs for Historical Knowledge
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"1j6sv": [{"id": "27937/9T2I7QLM", "source": "zotero"}], "2de4f": [{"id": "27937/F3XT4XAQ", "source": "zotero"}], "3fk4h": [{"id": "27937/KNEK45E4", "source": "zotero"}], "mptkr": [{"id": "27937/QD3X7XMD", "source": "zotero"}], "ors1l": [{"id": "27937/56EE9N63", "source": "zotero"}], "x9a26": [{"id": "27937/H9BUWE28", "source": "zotero"}]}} jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} -->
As historians explore the possibilities of generative AI, it is important to understand how these technologies are created and assessed. With this knowledge we can better evaluate their potential utility and their limits.

At the most fundamental level, generative AI models are statistical representations of the datasets on which they are trained. Machine learning techniques like deep learning and recent innovations like the Transformer network architecture (<cite id="1j6sv"><a href="#zotero%7C27937%2F9T2I7QLM">Vaswani et al., “Attention Is All You Need.”</a></cite>) have enabled the creation of models capable of mimicking the data on which they are trained with a high degree of fidelity. But researchers have also discovered that with sufficient time and the application of (often immense) computational power, these models exhibit a range of “emergent” capabilities. (<cite id="ors1l"><a href="#zotero%7C27937%2F56EE9N63">Wei et al., “Emergent Abilities of Large Language Models.”</a></cite>) For example, LLMs can summarize texts, perform language translation, write working computer code, and compose informative responses on a wide array of subjects - all without specific training on how to perform such tasks. (<cite id="3fk4h"><a href="#zotero%7C27937%2FKNEK45E4">Brown et al., “Language Models Are Few-Shot Learners.”</a></cite>) Moreover, these emergent capacities seem to "scale", meaning new models exhibit enhanced performance through training on ever-greater quantities of data and computation. (<cite id="x9a26"><a href="#zotero%7C27937%2FH9BUWE28">Kaplan et al., “Scaling Laws for Neural Language Models.”</a></cite>) The nature of these emergent capacities remains a matter of intense research and debate, as do the ethical and legal questions surrounding their use. However, it is clear that LLMs can both interpret and generate data in ways that rival previous machine learning methods. Scholars studying these AI systems have labeled them "foundational models" due to their potential to enable new domains of computational analysis. (<cite id="2de4f"><a href="#zotero%7C27937%2FF3XT4XAQ">Bommasani et al., “On the Opportunities and Risks of Foundation Models.”</a></cite>) Indeed, the remarkable versatility of LLMs is stimulating broader discussions about the potential implications of these technologies on society at large. (<cite id="mptkr"><a href="#zotero%7C27937%2FQD3X7XMD">Eloundou et al., “GPTs Are GPTs.”</a></cite>)
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} -->


<!-- #endregion -->

<!-- #region citation-manager={"citations": {"nppps": [{"id": "27937/U534FF7L", "source": "zotero"}], "r8lyr": [{"id": "27937/UYVGUT4C", "source": "zotero"}]}} jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["hermeneutics"] -->
While the Generative Pre-trained Transformer (GPT) series from OpenAI is the best known of these foundational models, there has been a rapid proliferation of commercial and open-source alternatives. Notable recent LLMs include Google’s Gemini, Anthropic’s Claude, and open-source models offered by Meta and Mistral.

Foundational models are also emerging in other domains, such as image, video, and audio synthesis. Architectures like CLIP (<cite id="r8lyr"><a href="#zotero%7C27937%2FUYVGUT4C">Radford et al., “Learning Transferable Visual Models From Natural Language Supervision.”</a></cite>) enable the creation of synthetic imagery in models like OpenAI’s DALL-E, Midjourney, and the open-source community behind Stable Diffusion. Similar approaches for generating video, speech, and music have been developed by firms like Runway-XL, ElevenLabs, and Suno, along with open-source alternatives hosted on sites likes HuggingFace. Most notably, new forms of LLM-training have enabled the combination of these capacities in multi-modal models capable of working across multiple domains, such as OpenAI’s GPT-4 series. (<cite id="nppps"><a href="#zotero%7C27937%2FU534FF7L">OpenAI, “GPT-4 Technical Report.”</a></cite>)

An accessible way to stay abreast of recent innovations in this field is by following the leaderboards used to measure performance on standard LLM benchmarks. [LLMArena’s Chatbot Arena](https://lmarena.ai/) offers an overview of leading contemporary models, while [HuggingFace’s Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) and the [Open Multilingual LLM Evaluation Leaderboard](https://huggingface.co/spaces/uonlp/open_multilingual_llm_leaderboard) offer specialized metrics for particular domains and use-cases. 
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"4lr4q": [{"id": "27937/FMW5DCWM", "source": "zotero"}], "e4unb": [{"id": "27937/EZNK3CE3", "source": "zotero"}], "g0kqk": [{"id": "27937/MVDFMR8K", "source": "zotero"}], "v35u4": [{"id": "27937/78DL3V96", "source": "zotero"}]}} jupyter={"outputs_hidden": false} -->
While such claims have sparked both excitement and alarm, any assessment of LLMs must first be tempered with humility. LLMs are often described as possessing “knowledge” and “understanding,” yet direct engagement with these models can quickly reveal both their remarkable breadth and their narrow limits. Incisive critics of this technology characterize LLMs as “stochastic parrots” that excel at uncanny mimicry of human intelligence. (<cite id="g0kqk"><a href="#zotero%7C27937%2FMVDFMR8K">Bender et al., “On the Dangers of Stochastic Parrots | Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency.”</a></cite>) A form of this mimicry has proven convincing in the past. The first attribution of true artificial intelligence to a computer program occurred in 1966 with a scripted chatbot named ELIZA, developed by AI pioneer Joseph Weizenbaum. (<cite id="e4unb"><a href="#zotero%7C27937%2FEZNK3CE3">McCorduck, <i>Machines Who Think a Personal Inquiry into the History and Prospects of Artificial Intelligence</i>.</a></cite>) A recent replication of this phenomenon occurred in June 2022 when a Google AI engineer declared the LLM he was training had become sentient. Such attributions will likely increase as newer LLMs demonstrate increasing proficiency in seemingly distinct human qualities, like humor. (<cite id="4lr4q"><a href="#zotero%7C27937%2FFMW5DCWM">Chowdhery et al., “PaLM.”</a></cite>) The means by which LLMs process, interpret, and generate information is a highly technical field requiring specialization in natural language processing, statistics, computational linguistics, and machine learning. While many historians may lack the technical knowledge to effectively evaluate the merits of these debates, when it comes to our own domain we are well equipped to offer informed insights.

Indeed, the standard measurement for a LLM’s historical knowledge was inadvertently created by historians. One widely-used measure for LLM performance is the Massive Multitask Language Understanding (MMLU) benchmark, developed in 2021 by researchers led by Dan Hendryks. This benchmark contains nearly 16,000 questions from 57 academic disciplines ranging in difficulty from an elementary educational level to postgraduate curricula in professional domains like law and medicine. History is measured in this benchmark through some six hundred questions taken from the Advanced Placement (A.P.) curricula for U.S., European, and World history. Hundreds of thousands of secondary students across the globe annually enroll in these curricula, which are designed to replicate the rigors of an introductory, university-level history course. The educators who developed and refined these programs likely never imagined their work would serve as a technical benchmark, and the appropriateness of such a standard can be debated. (<cite id="v35u4"><a href="#zotero%7C27937%2F78DL3V96">Marshall, “The Strange World of AP U.S. History.”</a></cite>) Yet this benchmark, however imperfect, offers historians an accessible means to evaluate this highly technical domain.
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} -->
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

<!-- #region citation-manager={"citations": {"wfmit": [{"id": "27937/ZS9JDNGD", "source": "zotero"}]}} jupyter={"outputs_hidden": false} -->
The MMLU benchmarks were first tested in 2021 against the then-leading LLM, OpenAI’s GPT-3. Twenty-five percent accuracy represented random chance; ninety percent performance reflected expert-level accuracy. GPT-3 initially achieved over fifty percent accuracy on all three A.P. curricula, and its performance in these subfields numbered among the top third of all the academic disciplines in the benchmarks. However, in no field did GPT-3 achieve expert-level accuracy, and the model demonstrated particularly poor performance in the fields of “Moral Questions” and “Professional Law.” As the authors note, this “weakness is particularly concerning because it will be important for future models to have a strong understanding of what is legal and what is ethical.” (<cite id="wfmit"><a href="#zotero%7C27937%2FZS9JDNGD">Hendrycks et al., “Measuring Massive Multitask Language Understanding.”</a></cite>)
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"tbd8o": [{"id": "27937/A834FRJL", "source": "zotero"}]}} jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["hermeneutics"] -->
The specific accuracy rates for GPT-3 for the initial Hendryks study: US History, 52.9%; European History, 53.9%; and World History, 56.1%. Full data for questions for history and other disciplines can be found at: (<cite id="tbd8o"><a href="#zotero%7C27937%2FA834FRJL">Hendrycks, <i>Measuring Massive Multitask Language Understanding</i>.</a></cite>) Many thanks to Dan Hendrycks for sharing the discipline-specific accuracy rates for these fields.
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} -->
However, rapid advances in model development have occurred since 2021. Subsequent tests on newer models “scaled” on ever greater amounts of data and computation demonstrate substantial gains in performance on these historical benchmarks. Below are results from a replication study conducted in September 2024 across a series of leading LLMs, along with the initial Hendryks test:
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

<!-- #region citation-manager={"citations": {"s177q": [{"id": "27937/5AL5LZ2K", "source": "zotero"}], "s4luc": [{"id": "27937/GSIXPJ7P", "source": "zotero"}]}} jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["hermeneutics"] -->
Data from this replication study can be accessed via the HELM Leaderboard for the MMLU Benchmark, hosted by the Center for Research on Foundation Models at Stanford University. (<cite id="s4luc"><a href="#zotero%7C27937%2FGSIXPJ7P">Mai and Liang, “Massive Multitask Language Understanding (MMLU) on HELM.”</a></cite>) You can directly experiment with LLM performance on these benchmarks via a digital history project accompanying this article, “What Do AIs Know About History?” (<cite id="s177q"><a href="#zotero%7C27937%2F5AL5LZ2K">Hutchinson.</a></cite>)
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"3unok": [{"id": "27937/YVTAGDKZ", "source": "zotero"}], "55r4n": [{"id": "27937/KNEK45E4", "source": "zotero"}], "5a9qa": [{"id": "27937/BD8996H7", "source": "zotero"}], "6cssb": [{"id": "27937/9GQG6VFM", "source": "zotero"}], "6ph9l": [{"id": "27937/5YDNQS4V", "source": "zotero"}], "8fjtz": [{"id": "27937/VEDFUUBA", "source": "zotero"}], "97pas": [{"id": "27937/X4D92B7V", "source": "zotero"}], "ahtmn": [{"id": "27937/BVBZMR66", "source": "zotero"}], "c6t3w": [{"id": "27937/TPGPSRAI", "source": "zotero"}], "fpott": [{"id": "27937/IEQ8GAVU", "source": "zotero"}], "j1kj5": [{"id": "27937/KNEK45E4", "source": "zotero"}], "jm1mt": [{"id": "27937/5GTQD5W9", "source": "zotero"}], "kba8r": [{"id": "27937/U534FF7L", "source": "zotero"}], "r1ql3": [{"id": "27937/TGPDB8WX", "source": "zotero"}], "vuott": [{"id": "27937/MVDFMR8K", "source": "zotero"}], "xndfm": [{"id": "27937/NYNDVYMM", "source": "zotero"}], "yh6j9": [{"id": "27937/S3ADX5DD", "source": "zotero"}], "zsn16": [{"id": "27937/MHRIEHH8", "source": "zotero"}]}} jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} -->
Rapid improvement on this benchmark have been made in just a few years, with a variety of commercial and open-source LLMs now demonstrating expert-level accuracy on all three of the subject exams. These findings mirror the striking performance of models like GPT-4 in other knowledge domains such as medical school curricula (<cite id="8fjtz"><a href="#zotero%7C27937%2FVEDFUUBA">Nori et al., “Capabilities of GPT-4 on Medical Challenge Problems.”</a></cite>), American bar exams, (<cite id="ahtmn"><a href="#zotero%7C27937%2FBVBZMR66">Katz, “GPT Takes the Bar Exam.”</a></cite>), and a host of other standardized assessments. (<cite id="kba8r"><a href="#zotero%7C27937%2FU534FF7L">OpenAI, “GPT-4 Technical Report.”</a></cite>)

Yet, why do some LLMs perform better in some knowledge domains than others? How can a model get one question right, while other questions generate errors? There is a temptation to parse the model’s performance in ways relatable to our human perspective. The human test taker might approach the question by assessing what types of historical thinking each question requires, what sort of knowledge is offered by the options, and how the historical source relates to the question. But, of course, LLMs aren’t human - and unlike the human test taker, these models have already seen the questions in advance. In 2022 alone, over 800,000 students took A.P. History exams. (<cite id="5a9qa"><a href="#zotero%7C27937%2FBD8996H7">“Program Summary Report.”</a></cite>) Significant online resources have emerged to serve the sizable population of students and instructors participating in this international curriculum. Hundreds of exam questions have migrated online via the collective efforts of the test prep publishing industry, various study apps, and uploaded example tests.

Thus the capabilities of LLMs on these benchmarks directly relates to the vast dataset used to train them: the Internet itself. The data collection built for training GPT-3 encompassed the majority of English-language Wikipedia, Reddit’s thousands of discussion forums, extensive corpora of digitized books, and the billions of web pages contained in the Common Crawl repository. (<cite id="55r4n"><a href="#zotero%7C27937%2FKNEK45E4">Brown et al., “Language Models Are Few-Shot Learners.”</a></cite>) The training sets used for subsequent LLMs remains largely unknown, as AI firms keep their data a closely guarded and proprietary asset; indeed, the future of LLMs may depend on pending litigation concerning copyright infringement in the use of this data. Yet given the scale of such datasets, many of the A.P. History questions used in these benchmarks have likely ended up in LLM training data. If those who critique LLMs as “stochastic parrots” are correct, these gains in performance come from improvements in models memorizing this data, and not through any analytical process. (<cite id="vuott"><a href="#zotero%7C27937%2FMVDFMR8K">Bender et al., “On the Dangers of Stochastic Parrots | Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency.”</a></cite>) When LLMs encounter questions outside of their training, their accuracy is likely to suffer.

Yet such inaccuracies can be difficult to detect. LLMs tend to confidently assert error as fact, a phenomenon described by AI researchers as “hallucinations.” Such hallucinations represent a major challenge in LLM research and for many practical applications of this technology, particularly given the remarkable effectiveness of these models in generating convincing and otherwise accurate prose. (<cite id="6cssb"><a href="#zotero%7C27937%2F9GQG6VFM">Ji et al., “Survey of Hallucination in Natural Language Generation.”</a></cite>)  Initial testing by OpenAI on the GPT series demonstrated that human readers often struggle to identify text generated by LLMs. (<cite id="j1kj5"><a href="#zotero%7C27937%2FKNEK45E4">Brown et al., “Language Models Are Few-Shot Learners.”</a></cite>) Rectifying such hallucinations is a significant area of LLM research. However, some scholars, like computational linguist Emily Bender, argue that such behaviors are inherent flaws in LLMs. (<cite id="c6t3w"><a href="#zotero%7C27937%2FTPGPSRAI">Bender, “On NYT Magazine on AI.”</a></cite>)

Additional risks confront historians using these technologies. While AI firms seek to remove potentially offensive texts from their training sets, the sheer scale of this data make selective curation very challenging. LLMs thus generate responses reflecting both the best and the worst of our online world. This reality has troubled previous AI implementations. Well-intentioned researchers have created chatbots that spew hateful invective, human resources applications that refuse to hire female applicants, and algorithms based on criminal justice sentencing guidelines that starkly reinforce racial disparities already prevalent in the carceral system. (<cite id="6ph9l"><a href="#zotero%7C27937%2F5YDNQS4V">Barton, “Algorithmic Bias Detection and Mitigation.”</a></cite>) Early models in the GPT series have been known to unexpectedly generate responses in innocuous contexts containing violent imagery, sexually explicit language, and racial, ethnic, and religious slurs. (<cite id="xndfm"><a href="#zotero%7C27937%2FNYNDVYMM">Strickland, “OpenAI’s GPT-3 Speaks! (Kindly Disregard Toxic Language) - IEEE Spectrum.”</a></cite>) These findings further confirm the prescient warnings offered by scholars such as Safiya Umoja Noble (<cite id="fpott"><a href="#zotero%7C27937%2FIEQ8GAVU">Noble, <i>Algorithms of Oppression</i>.</a></cite>), Timnit Gebru (<cite id="yh6j9"><a href="#zotero%7C27937%2FS3ADX5DD">Gebru, “Race and Gender.”</a></cite>), Ruha Benjamin (<cite id="97pas"><a href="#zotero%7C27937%2FX4D92B7V">Benjamin, <i>Race After Technology</i>.</a></cite>), Kate Crawford (<cite id="3unok"><a href="#zotero%7C27937%2FYVTAGDKZ">Crawford, <i>Atlas of AI</i>.</a></cite>), and Trevor Paglen (<cite id="jm1mt"><a href="#zotero%7C27937%2F5GTQD5W9">Crawford and Paglen, “Excavating AI.”</a></cite>) on digital practices that reinforce analog inequalities. Some AI researchers consider such behaviors as lamentable but solvable problems through further technical advances, particularly with the use of methods like Reinforcement Learning from Human Feedback (RLHF). (<cite id="r1ql3"><a href="#zotero%7C27937%2FTGPDB8WX">Christiano et al., “Deep Reinforcement Learning from Human Preferences.”</a></cite>) Reducing the impact of such biases is a significant research area, particularly through the creation of smaller, more carefully curated datasets for AI training. However, many historians will likely share the skepticism of some researchers concerning such mitigations. (<cite id="zsn16"><a href="#zotero%7C27937%2FMHRIEHH8">Gehman et al., “RealToxicityPrompts.”</a></cite>) Bias emerges from more than just explicit language or imagery but from the very structures of societies. Can any historical source be separated from its context as a neutral artifact, free of its creator’s perspective and the influences of its time? What about the untold millions of sources that make up the scale of an LLM’s training set?

To be sure, LLMs are imperfect digital tools, and given these flaws historians must exercise caution when employing this technology. Yet scholars are finding that within the confines of these imperfections there is real potential to advance historical research. While a LLM’s facility with multiple-choice questions might be the product of memorization, such knowledge has long been a springboard for more advanced forms of inquiry. And A.P. study guides are not the only historical texts LLMs are trained on. Primary source collections, academic monographs, open-source scholarly journals - these too inform an LLM’s training. 

The influence of these sources can be found when LLMs are posed more complex questions in a structured prompt. Let’s return to the earlier A.P. question above featuring the Populist-era campaign song “The Hayseed.” In the code blocks below, GPT-4 is given the lyrics and publication history of the song. GPT-4 is then prompted to identify the larger historical context of the source, the song’s intended purpose and audience, and how the source might be interpreted via different historiographical approaches.
<!-- #endregion -->

```python jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["figure-hayseed-*"]
from IPython.display import Image, display

metadata={
    "jdh": {
        "object": {
            "type":"image",
            "source": [
                "Arthur L. Kellog, “The Hayseed,” *Farmers Alliance* (4 October 1890). Nebraska Newspapers (University of Nebraska Libraries), https://nebnewspapers.unl.edu/lccn/2017270209/1890-10-04/ed-1/seq-1/. \n\nOriginal citation found in: John Donald Hicks, *The Populist Revolt: A History of the Farmers' Alliance and the People's Party* (University of Minnesota Press, 1931), 168, fn. 30."
            ]
        }
    }
}


# Display the image
display(Image('./media/hayseed.png', width=600), metadata=metadata)



```

```python jupyter={"outputs_hidden": false}
import requests




# URL of the raw text file on GitHub
file_url = 'https://raw.githubusercontent.com/Dr-Hutchinson/jdh_submission/refs/heads/main/media/prompts/primary_source_analysis.txt'

# Fetch the content of the file
response = requests.get(file_url)

# Format the prompt text for primary source analysis
primary_source_analysis_prompt = response.text.replace('\\n', '\n')
display(Markdown("**Primary Source Analysis Prompt fed to GPT-4o:**\n\n" + primary_source_analysis_prompt))


```

```python jupyter={"outputs_hidden": false}
# installing libraries
!pip install openai
!pip install jiwer
!pip install rich
```

```python jupyter={"outputs_hidden": false}
# Enter OpenAI API key in the space below.
# Access to OpenAI's API keys can be found here: https://beta.openai.com/signup

import os
os.environ["OPENAI_API_KEY"] = ""
```

```python jupyter={"outputs_hidden": false}
# Code for running primary source analysis of the "Hayseed" with OpenAI's GPT-4o model.

from openai import OpenAI
from rich.console import Console

# Initialize the console for rich output
console = Console()

# Initialize the OpenAI client
client = OpenAI()

# Create the query for the LLM
query = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": primary_source_analysis_prompt}
    ]
)

# Extract the output from the response
output = query.choices[0].message.content

# Create formatted text for display
output_text = (
    "[bold]GPT-4's Interpretation of 'The Hayseed':[/bold]\n\n"
    f"{output}\n\n"
    "[bold]End of GPT-4's Interpretation[/bold]"
)

# Display the formatted output
console.print(output_text, width=console.size.width)

```

<!-- #region citation-manager={"citations": {"rxaxc": [{"id": "27937/G5ESJ8NI", "source": "zotero"}]}} jupyter={"outputs_hidden": false} -->
While one can debate aspects of GPT-4’s interpretations, it nonetheless accurately captures much of the context and intent of the source. With the right design, LLMs could be automated to annotate an entire corpus of archival sources in a similar manner, becoming a tool of the digital historian overwhelmed by an abundance of historical data, as envisioned by Roy Rozenweig twenty years ago. Yet LLM outputs and hallucinations are already contributing to this deluge of data. Both benign and malicious use of these technologies are impacting our understanding of the past and ability to comprehend the present. Historians should contribute to the broader dialogue about the implications and informed use of these technologies, especially as they become increasingly embedded in our digital lives. Further experimentation is also needed to more fully assess LLM’s capabilities for historical interpretation, as well as the creation of new benchmarks to test different approaches to historical analysis. But progress moves quickly in the field of generative AI, and there is intense competition to build new models that advance the existing capabilities of LLMs while shedding their shortcomings. Yet progress remains uneven. Of significant concern are LLM’s performance on benchmarks on ethics and morality, which continue to demonstrate troubling areas of weakness. (<cite id="rxaxc"><a href="#zotero%7C27937%2FG5ESJ8NI">Hoffmann et al., “Training Compute-Optimal Large Language Models.”</a></cite>)

While imperfect tools, their flaws do not mean that LLMs have no place in the historian’s toolkit. In fact, by acknowledging and confronting these shortcomings, historians can better contribute our disciplinary perspectives on the debates concerning this technology, particularly in leveraging the strengths of these models to empower and broaden accessibility to historical sources. The case studies below demonstrate how historians are using LLMs as a versatile tools for both researching and communicating the past.
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} -->
# Prompting LLMs for Digital History: Case Studies
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"c1hh3": [{"id": "27937/XEUKQDPE", "source": "zotero"}], "zf8ru": [{"id": "27937/VXGSAGTI", "source": "zotero"}]}} jupyter={"outputs_hidden": false} -->
A promising approach for LLMs and other foundational AI models lies in their capacity to assist data preparation and cleanup – a process that often constitutes an estimated 80% of the labor involved in preparing data for analysis. (<cite id="zf8ru"><a href="#zotero%7C27937%2FVXGSAGTI">Dasu and Johnson, <i>Exploratory Data Mining and Data Cleaning</i>.</a></cite>) Digitized historical materials frequently require transcriptions, error correction, and the creation of extensive metadata. These essential but time-consuming tasks can become research bottlenecks. However, through use of simple prompting techniques historians can leverage the power of LLMs to streamline and accelerate the creation of “tidy datasets” possessing standardized and ordered structures essential for analysis and replication. (<cite id="c1hh3"><a href="#zotero%7C27937%2FXEUKQDPE">Wickham, “Tidy Data.”</a></cite>)
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} -->
## Case Study: Oral History Transcriptions
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"d9rvh": [{"id": "27937/7VHKCH3M", "source": "zotero"}], "etf8p": [{"id": "27937/YWJAQ4V8", "source": "zotero"}]}} jupyter={"outputs_hidden": false} -->
Oral history provides a particularly useful case study for demonstrating the potential utility of generative AI. Transcribing audio recordings is a central activity in this methodology, a task typically requiring significant time and labor. As one oral history guide notes, transcribing a single hour typically requires six to eight hours of manual processing and review. (<cite id="etf8p"><a href="#zotero%7C27937%2FYWJAQ4V8">Ritchie, <i>Doing Oral History</i>.</a></cite>) However, advances in specialized generative AI models permits significant streamlining of this task.

Notable among these models is Whisper, an open-source audio transcription and translation model developed by OpenAI that belongs to the same Transformer family as the GPT series. (<cite id="d9rvh"><a href="#zotero%7C27937%2F7VHKCH3M">Radford et al., “Robust Speech Recognition via Large-Scale Weak Supervision.”</a></cite>) Trained on over 180,000 hours of audio recordings, this model demonstrates performance comparable (and in some cases exceeding) the accuracy rates of human transcriptions in over 57 languages. In this test we’ll examine Whisper’s performance on the first two minutes of a transcribed oral history of historian John Hope Franklin by the Southern Oral History Program. Recorded on audiotape in 1990, this segment features multiple voices, crosstalk, filler words, and background noise - typical features for many oral history recordings, but features that nonetheless complicate efforts to create accurate transcripts. In the code below, we will use Whisper to transcribe the audio segment and compare it against the official transcript. 
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"kav0z": [{"id": "27937/I2BKP7MN", "source": "zotero"}]}} jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["hermeneutics"] -->
The Whisper series is offered as a series of open-source voice recognition and voice translation models across several tiers of computing power and freely available on sites like [HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/whisper). However, for simplicity this demonstration code uses OpenAI’s API for the Whisper-2-large model. As of September 2024, OpenAI charged $0.36 per hour of recorded time for transcriptions using the API.

For a detailed and informative tutorial on using and analyzing Whisper, see: (<cite id="kav0z"><a href="#zotero%7C27937%2FI2BKP7MN">Schultz, “[Tutorial] Using Whisper to Transcribe Oral Interviews – CSS @ IPP.”</a></cite>)
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
audio_url = "https://github.com/Dr-Hutchinson/jdh_submission/raw/refs/heads/main/media/A-0339_edited.mp3"

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
audio_length_seconds = 153  # 2 minutes and 33 seconds in seconds
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
from rich.console import Console

# Initialize the console for rich output
console = Console()

# URL for the audio file on GitHub
audio_url = "https://github.com/Dr-Hutchinson/jdh_submission/raw/refs/heads/main/media/A-0339_edited.mp3"

# Save location for the downloaded audio file
file_path = "./A-0339_edited.mp3"

# Displaying citation
citation_text = (
    "[bold]Citation:[/bold]\n"
    "[italic]“John Hope Franklin and John Egerton, Conducted by Oral History Interview with John Hope Franklin, "
    "July 27, 1990. Interview A-0339. Southern Oral History Program Collection (#4007).”[/italic]\n"
    "[dim]https://docsouth.unc.edu/sohp/A-0339/menu.html[/dim]"
)

# Print the citation with console output
console.print(citation_text, width=console.size.width)

# Load and play the saved audio file
Audio(file_path)
```

<!-- #region jupyter={"outputs_hidden": false} -->
Based on the professional standard, this excerpt would take approximately fifteen to twenty minutes to manually transcribe. Whisper achieved this in less than ten seconds.

How accurate is the model compared to a human-produced transcript? Due to the stochastic nature of these models, each time you run this code slightly different variations might occur, particularly in the most challenging segments. The code block below visualizes a sample transcription produced by Whisper that was annotated and compared against the original. Notable omissions and discrepancies are highlighted. Whisper’s accuracy is then calculated via a standard benchmark for audio transcription, the word error rate (WER).
<!-- #endregion -->

```python jupyter={"outputs_hidden": false} slideshow={"slide_type": ""}
import requests
import re
from jiwer import wer
from IPython.display import display, HTML

# Function to clean HTML tags
def clean_html(text):
    return re.sub(r'<.*?>', '', text)

# URLs for the transcripts on GitHub
original_file_url = "https://raw.githubusercontent.com/Dr-Hutchinson/jdh_submission/refs/heads/main/media/revised_original_transcript_formatted.txt"
whisper_file_url = "https://raw.githubusercontent.com/Dr-Hutchinson/jdh_submission/refs/heads/main/media/revised_whisper_transcript_formatted.txt"

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

<!-- #region citation-manager={"citations": {"4y95j": [{"id": "27937/MYFQUX4C", "source": "zotero"}], "7mtvf": [{"id": "27937/KKDPZJYW", "source": "zotero"}], "a22ms": [{"id": "27937/4ITT4MQK", "source": "zotero"}], "yq0ts": [{"id": "27937/UHZYQM3W", "source": "zotero"}]}} jupyter={"outputs_hidden": false} -->
There are some suggestive observations we can take from these results. Closer inspection of the Whisper transcript shows some errors, a significant omission, differences in syntax, and literal transcriptions that an edited transcript would likely leave out. But given the media format and its audio quality, the oral historian has a solid first draft in just a few seconds. While the WER score indicates a need for final human review, that review will take considerably less effort and enable oral historians to shift their focus to interpreting, annotating, and validating their transcriptions. And even the best human transcriptions still contains errors. Take note of the final paragraph in the original transcript, which names Harvard as the destination of E. Franklin Frazier in 1934; but the noted sociologist actually joined the faculty of Howard University. Here Whisper accurately corrects a human error in the transcription. 

Applications like Whisper are already changing the field of oral history, as well as journalism, court reporting, and language translation. (<cite id="yq0ts"><a href="#zotero%7C27937%2FUHZYQM3W">Somers, “Whispers of A.I.’s Modular Future | The New Yorker.”</a></cite>) Scholars are using these techniques to complete multi-lingual transcriptions of aging and vulnerable media (<cite id="7mtvf"><a href="#zotero%7C27937%2FKKDPZJYW">Lehečka et al., “Transformer-Based Speech Recognition Models for Oral History Archives in English, German, and Czech.”</a></cite>) while also enabling new forms of community-based scholarship and teaching. (<cite id="4y95j"><a href="#zotero%7C27937%2FMYFQUX4C">Rochester Institute of Technology, “Artificial Intelligence Aids Cultural Heritage Researchers Documenting and Teaching Oral Histories.”</a></cite>) Like other forms of generative AI, Whisper is prone to bias, error, and hallucination. (<cite id="a22ms"><a href="#zotero%7C27937%2F4ITT4MQK">Koenecke et al., “Careless Whisper.”</a></cite>) Scholars should use this model to augment human review, and not replace it.

Generative AI is being used to process and transform other media types as well. Just as Whisper transforms spoken language into text, LLMs can be leveraged to process visual information, such as printed or handwritten text, to aid in source digitization.
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} -->
## Case Study: Error Correction of Optical Character Recognition Scans
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"0jirx": [{"id": "27937/ZJW9AI49", "source": "zotero"}], "b6864": [{"id": "27937/TIAJYHF6", "source": "zotero"}]}} jupyter={"outputs_hidden": false} -->
Another potential use case for AI models in digital history is error correction of optical character recognition (OCR) scans. Machine learning techniques, such as those pioneered by the research team at Transkribus, have greatly enhanced the quality, speed, and cost-effectiveness of OCR for a broad range of historical texts. (<cite id="b6864"><a href="#zotero%7C27937%2FTIAJYHF6">Muehlberger et al., “Transforming Scholarship in the Archives through Handwritten Text Recognition.”</a></cite>) However, even high-fidelity OCR scans often produce errors that impact the accessibility and searchability of text collections. (<cite id="0jirx"><a href="#zotero%7C27937%2FZJW9AI49">Milligan, “Illusionary Order.”</a></cite>) In the examples below, we can observe how a LLM can be prompted to correct these errors.

The following image is from a newspaper published in a German prisoner-of-war camp in Mississippi during World War II and later microfilmed by the Library of Congress.
<!-- #endregion -->

```python jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["figure-lotse-6-30-1945-*"]
from PIL import Image
from IPython.display import display, Image as IPImage
from rich.console import Console

# Initialize the console for rich output
console = Console()

# Load and resize the image
ocr2_url = 'https://raw.githubusercontent.com/Dr-Hutchinson/jdh_submission/refs/heads/main/media/die_lotse_6-30-45_1.png'
image2 = Image.open('./die_lotse_6-30-45_1.png')

# Set new dimensions for the resized image
new_width = 600
new_height = int(image2.height * (new_width / image2.width))

# Resize the image for better visualization
resized_image = image2.resize((new_width, new_height), Image.LANCZOS)

# Convert the PIL image to a format compatible with IPython display 
resized_image.save("/tmp/resized_image.png")  
display(IPImage(filename="/tmp/resized_image.png")) 

# Format the citation text using rich
citation_text = (
    "[bold]Source:[/bold] [italic]“Nur ein Film?,”[/italic] [italic]Die Lotse[/italic] (Camp McCain, Mississippi), 30 June [not bold]1945[/not bold].\n"
    "In: Karl John Richard Arndt, editor. [italic]German P.O.W. Camp Papers[/italic]. (Washington, D.C.: Library of Congress Photoduplication Service, [not bold]1965)[/not bold]. Reel [not bold]9[/not bold]."
)

# Display the formatted citation
console.print(citation_text)
```

<!-- #region citation-manager={"citations": {"5zf5d": [{"id": "27937/KNEK45E4", "source": "zotero"}]}} jupyter={"outputs_hidden": false} -->
Let’s compare a human transcription of this image against outputs generated by Google’s Cloud Vision OCR service and GPT-4. In this code block GPT-4 is given the raw OCR output along with a [prompt](https://raw.githubusercontent.com/Dr-Hutchinson/jdh_submission/refs/heads/main/media/prompts/ocr_prompt.txt) for correcting OCR errors. This prompt includes examples of the task we wish the model to perform and is tailored to the type of data the model will encounter. This method, called few-shot prompting, is a common “prompt engineering” method for effectively guiding LLM generations. (<cite id="5zf5d"><a href="#zotero%7C27937%2FKNEK45E4">Brown et al., “Language Models Are Few-Shot Learners.”</a></cite>) By again using the word error rate (WER), we can compare the relative accuracy of these different techniques against human performance. Discrepancies between the human-created transcription and the OCR outputs are highlighted.
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"s9wzk": [{"id": "27937/5G5LJCLC", "source": "zotero"}], "t1kfr": [{"id": "27937/3H7M3AJ8", "source": "zotero"}]}} jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["hermeneutics"] -->
Effectively utilizing LLMs for historical research extends beyond simply inputting text instructions; it demands an understanding of how to guide these models toward desired outcomes. This process, known as prompt engineering, has emerged as a critical skill for leveraging the power of LLMs. By carefully structuring instructions and providing relevant context, historians can shape LLM outputs to address a wide array of research needs.

Prompt engineering is an iterative process. Experimentation is often required, using different prompt structures tailored to the dataset to achieve the desired outputs. One prominent prompt engineering technique is few-shot prompting. This method involves providing the LLM with a small number of examples demonstrating a specific task and desired output format. An emergent ability of LLMs called “in-context learning” allows LLMs to adapt their approach based on a few carefully chosen examples. Such examples can significantly improve the model’s ability to generalize to new, unseen data. This technique is especially valuable for tasks where explicit rules are difficult to define, allowing the LLM to learn from demonstration rather than strict programming. 

Few-shot prompting is among the most common of a rapidly growing number of prompt approaches. (<cite id="t1kfr"><a href="#zotero%7C27937%2F3H7M3AJ8">Vatsal and Dubey, “A Survey of Prompt Engineering Methods in Large Language Models for Different NLP Tasks.”</a></cite>) Regardless of the technique, effective prompt prompting for historical research does not require extensive technical skill, but instead benefits from clear communication and the application of specific domain knowledge. As LLMs become increasingly integrated into digital historical practice, the ability to craft effective prompts may join skills like text analysis and data visualization as useful components of the digital historian’s toolkit.

To learn more about the practices of prompt engineering, a good starting place is DAIR.AI’s Prompt Engineering Guide, which lays out accessible examples for various prompting approaches. (<cite id="s9wzk"><a href="#zotero%7C27937%2F5G5LJCLC">Saravia, <i>Prompt Engineering Guide</i>.</a></cite>)
<!-- #endregion -->

```python jupyter={"outputs_hidden": false}
import requests
import difflib
import re
from jiwer import wer
from IPython.display import display, HTML
from openai import OpenAI

# Function to fetch text content from a URL
def fetch_text_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful
    return response.text.strip()

# Function to annotate differences between two texts
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

# Function to calculate Word Error Rate (WER)
def calculate_wer(original_text, compared_text):
    error_rate = wer(original_text, compared_text)
    return f'{error_rate:.2%}'

# Function to display annotated transcripts side by side
def display_side_by_side(ocr_output, corrected_output, title_ocr, title_corrected):
    # Use difflib to identify differences
    differ = difflib.Differ()
    diff = list(differ.compare(ocr_output.split(), corrected_output.split()))
    
    # Create annotated versions of the transcripts
    ocr_annotated = annotate_differences(diff, target='ocr')
    corrected_annotated = annotate_differences(diff, target='human')
    
    # Construct HTML to display the two annotated versions side by side
    html_content = f'''
    <div style="display: flex; align-items: flex-start;">
        <div style="width: 50%; padding-right: 20px; border-right: 1px solid black;">
            <h4>{title_ocr}:</h4>
            <div>{ocr_annotated}</div>
        </div>
        <div style="width: 50%; padding-left: 20px;">
            <h4>{title_corrected}:</h4>
            <div>{corrected_annotated}</div>
        </div>
    </div>
    '''
    
    # Display the HTML content in Jupyter
    display(HTML(html_content))

# Function to run the whole comparison workflow
def run_comparison(ocr_output, corrected_output, title_ocr, title_corrected):
    # Display side-by-side comparison
    display_side_by_side(ocr_output, corrected_output, title_ocr, title_corrected)
    
    # Create a clean title for WER by removing any text in parentheses
    wer_title_corrected = title_corrected.split('(')[0].strip()
    
    # Calculate and display WER
    error_rate = calculate_wer(corrected_output, ocr_output)
    display(HTML(f'<h4>Word Error Rate (WER) for {wer_title_corrected}: {error_rate}</h4>'))

# Function to query GPT-4 for OCR corrections
def query_gpt4(ocr_prompt, ocr_output):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": ocr_prompt + "\n" + ocr_output}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)  

# URLs for the OCR outputs, human corrections, and prompt for GPT-4
file_urls = {
    "ocr_1": "https://raw.githubusercontent.com/Dr-Hutchinson/jdh_submission/refs/heads/main/media/die_lotse_1_ocr_output.txt",
    "corrected_1": "https://raw.githubusercontent.com/Dr-Hutchinson/jdh_submission/refs/heads/main/media/die_lotse_1_human_correction.txt",
    "ocr_2": "https://raw.githubusercontent.com/Dr-Hutchinson/jdh_submission/refs/heads/main/media/die_lotse_2_ocr_output.txt",
    "corrected_2": "https://raw.githubusercontent.com/Dr-Hutchinson/jdh_submission/refs/heads/main/media/die_lotse_2_human_correction.txt",
    "ocr_prompt": "https://raw.githubusercontent.com/Dr-Hutchinson/jdh_submission/refs/heads/main/media/prompts/ocr_prompt.txt"
}

# Initialize the OpenAI client
client = OpenAI()

# Load content from URLs
ocr_output_1 = fetch_text_from_url(file_urls["ocr_1"])
human_corrected_output_1 = fetch_text_from_url(file_urls["corrected_1"])
ocr_prompt = fetch_text_from_url(file_urls["ocr_prompt"])

# Make API Call for GPT-4 Correction of OCR Output
gpt4_corrected_output_1 = query_gpt4(ocr_prompt, ocr_output_1)

# Block 1: Human-corrected transcript vs OCR output
run_comparison(
    ocr_output_1,
    human_corrected_output_1,
    "Human Corrected Transcript (Corrections in green)",
    "OCR Transcript (Errors in red)",
)

# Block 2: Human-corrected transcript vs GPT-4 output
run_comparison(
    gpt4_corrected_output_1,
    human_corrected_output_1,
    "Human Corrected Transcript (Corrections in green)",
    "GPT-4 Corrected Transcript (Errors in red)"
)
```

<!-- #region jupyter={"outputs_hidden": false} -->
While the image quality is satisfactory and the text is printed using modern typefaces, the OCR scan still generates errors requiring human correction. Correcting even minor errors necessitates review, representing significant labor when processing a sizable text corpus. The LLM accelerates that task in this case by correcting OCR errors ahead of human review, particularly when guided by detailed instructions and a few examples tailored to the dataset or OCR task.

However, there are limits to this prompt engineering technique. Accuracy falls for both OCR models and LLMs alike when processing images containing considerable ‘noise’ and distortion, as in the image below. But recent LLMs like GPT-4 have been trained on multi-modal data, allowing them to process images as well as text. In the code below, GPT-4 is fed a specialized [prompt](https://raw.githubusercontent.com/Dr-Hutchinson/jdh_submission/refs/heads/main/media/prompts/gpt_vision_prompt.txt), a [few examples](https://raw.githubusercontent.com/Dr-Hutchinson/jdh_submission/refs/heads/main/media/prompts/vision_few_shot.txt), the raw OCR output, as well as the original image to help guide the model in correcting OCR errors.
<!-- #endregion -->

```python jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["figure-lotse-3-15-1945-*"]
from PIL import Image
from IPython.display import display, Image as IPImage
from rich.console import Console

# Initialize the console for rich output
console = Console()

# Load and resize the image
ocr2_url = 'https://raw.githubusercontent.com/Dr-Hutchinson/jdh_submission/refs/heads/main/media/die_lotse_3-15-45_1.png'
image2 = Image.open('./die_lotse_3-15-45_1.png')

# Set new dimensions for the resized image
new_width = 600
new_height = int(image2.height * (new_width / image2.width))

# Resize the image for better visualization
resized_image = image2.resize((new_width, new_height), Image.LANCZOS)

# Prepare the image for IPython display
resized_image.save("/tmp/resized_image.png")  
display(IPImage(filename="/tmp/resized_image.png")) 

# Create formatted citation text with rich
citation_text = (
    "[bold]Source:[/bold] [italic]“Zum Geleit,”[/italic] [italic]*Die Lotse*[/italic] (Camp McCain, Mississippi), 15 March [not bold]1945[/not bold].\n"
    "In: Karl John Richard Arndt, editor. [italic]*German P.O.W. Camp Papers*[/italic]. (Washington, D.C.: Library of Congress Photoduplication Service, [not bold]1965)[/not bold]. Reel [not bold]9[/not bold]."
)

# Display the formatted citation
console.print(citation_text)

```

```python jupyter={"outputs_hidden": false}
import os
import base64
import requests
from PIL import Image
from IPython.display import display, HTML, Markdown

# Function to fetch text content from a URL
def fetch_text_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  
    return response.text.strip()

# Function to encode the image to base64 for API call
def encode_image_from_url(image_url):
    response = requests.get(image_url)
    response.raise_for_status()
    return base64.b64encode(response.content).decode('utf-8')

# Function to prepare the payload for the API request, with OCR output and vision_few_shot included
def prepare_payload(encoded_image, prompt, ocr_output, image_filename):
    # Combine gpt_vision_prompt, vision_few_shot, and OCR output into a single prompt
    full_prompt = f"{gpt_vision_prompt}\n\n{vision_few_shot}\n\nOCR Output:\n{ocr_output}\n\nPlease correct the OCR errors for the image {image_filename}."
    
    return {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        "max_tokens": 1000
    }

# Function to make an API call to GPT-4 for OCR correction
def get_gpt4_ocr_output(payload):
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_json = response.json()
    if 'choices' in response_json:
        return response_json['choices'][0]['message']['content']
    else:
        print("Error in OCR:", response_json)
        return ""

# URLs for the image, human-corrected transcription, OCR output, and prompts for GPT-4
file_urls = {
    "image_2": "https://raw.githubusercontent.com/Dr-Hutchinson/jdh_submission/refs/heads/main/media/die_lotse_3-15-45_1.png",
    "corrected_2": "https://raw.githubusercontent.com/Dr-Hutchinson/jdh_submission/refs/heads/main/media/die_lotse_2_human_correction.txt",
    "ocr_2": "https://raw.githubusercontent.com/Dr-Hutchinson/jdh_submission/refs/heads/main/media/die_lotse_2_ocr_output.txt",
    "gpt_vision_prompt": "https://raw.githubusercontent.com/Dr-Hutchinson/jdh_submission/refs/heads/main/media/prompts/gpt_vision_prompt.txt",
    "vision_few_shot": "https://raw.githubusercontent.com/Dr-Hutchinson/jdh_submission/refs/heads/main/media/prompts/vision_few_shot.txt"
}

# Load the human-corrected transcript and OCR output for image from URLs
human_corrected_output = fetch_text_from_url(file_urls["corrected_2"])
ocr_output = fetch_text_from_url(file_urls["ocr_2"])

# Load gpt_vision_prompt and few shot examples from URLs
gpt_vision_prompt = fetch_text_from_url(file_urls["gpt_vision_prompt"])
vision_few_shot = fetch_text_from_url(file_urls["vision_few_shot"])

# Encode image from the URL
encoded_image = encode_image_from_url(file_urls["image_2"])

# Load the API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Set up headers for the API request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Prepare the payload and get GPT-4 OCR output for image
image_filename = os.path.basename(file_urls["image_2"])  
payload = prepare_payload(encoded_image, gpt_vision_prompt, ocr_output, image_filename)  
gpt4_ocr_output = get_gpt4_ocr_output(payload)

# Run comparisons for image_2

# Block 1: Human vs OCR
run_comparison(
    ocr_output,
    human_corrected_output,
    "Human Corrected Transcript (Corrections in green)",
    "OCR Transcript (Errors highlighted in red)"
)

# Block 2: Human vs GPT-4 Vision
run_comparison(
    gpt4_ocr_output,
    human_corrected_output,
    "Human Corrected Transcript (Corrections in green)",
    "GPT-4 Corrected Transcript (Errors highlighted in red)"
)
```

<!-- #region citation-manager={"citations": {"27nt7": [{"id": "27937/ZXTQBIJU", "source": "zotero"}], "6jdd5": [{"id": "27937/JV9GGCQA", "source": "zotero"}], "t1nqo": [{"id": "27937/58X69RSW", "source": "zotero"}]}} jupyter={"outputs_hidden": false} -->
While GPT-4 provides a less accurate transcription for this difficult and degraded image, it did significantly correct and improve the initial OCR output. There are other means to achieve even better performance. Specialized LLMs trained specifically for OCR tasks, like Microsoft’s Phi model series, show great promise in advancing the fidelity of such corrections. (<cite id="t1nqo"><a href="#zotero%7C27937%2F58X69RSW">Abdin et al., “Phi-3 Technical Report.”</a></cite>) Similarly, Pleias, a French AI research group, has been applying LLMs at scale for post-OCR correction in larger text datasets such as the Common Corpus, which comprises 500 billion words of text scraped from the Internet.  Their recent release of a multilingual, billion-word dataset of corrected OCR text includes English, French, German, and Italian texts from cultural heritage repositories such as Gallica and Chronicling America. (<cite id="27nt7"><a href="#zotero%7C27937%2FZXTQBIJU">PleIAs, “PleIAs/Post-OCR-Correction · Datasets at Hugging Face.”</a></cite>) However, while the quality of these corrections has shown significant improvement, Pleias’ work also highlights potential limitations of LLM-based OCR correction. Early tests revealed issues such as language-switching (e.g., parts of English texts mistakenly corrected into French or German) and the risk of hallucinations. (<cite id="6jdd5"><a href="#zotero%7C27937%2FJV9GGCQA">Langlais, “Post-OCR-Correction.”</a></cite>) Despite these limitations, the potential for LLMs in post-OCR correction is significant. As LLMs continue to evolve, their capacity to perform tasks like OCR correction will likely improve, but human review will remain essential to ensure accuracy.

These two case studies demonstrate LLMs' capacity to assist in various forms of data cleanup and preparation. While human review remains essential, LLMs can make that review less time-consuming and labor-intensive. Such approaches can improve the accuracy, lower the costs, and accelerate the pace of data preparation. The same is true for data extraction. In the following example, a LLM will demonstrate how such data can be captured by use of guided prompts.
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} -->
## Case Study: Structured Data Extraction
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"3y4ir": [{"id": "27937/HIPL38QS", "source": "zotero"}], "6161c": [{"id": "27937/68YHDUH6", "source": "zotero"}]}} jupyter={"outputs_hidden": false} -->
Automated data extraction has increasingly become an important part of digital scholarship. Reseachers have utilized a variety of computational approaches for compiling ordered data from historical media. Named entity recognition (NER) is one such example, a method which uses machine learning models like [spaCy](https://spacy.io/usage/facts-figures) to extract locations, events, individuals, and even concepts from texts into ordered classifications. Such techniques have enabled the creation of rich metadata from archival collections (<cite id="3y4ir"><a href="#zotero%7C27937%2FHIPL38QS">Chastang, Aguilar, and Tannier, “A Named Entity Recognition Model for Medieval Latin Charters.”</a></cite>) and museum catalogs. (<cite id="6161c"><a href="#zotero%7C27937%2F68YHDUH6">Nikolova and Levy, “Using Named Entity Recognition to Enhance Access to a Museum Catalog – Document Blog.”</a></cite>) Similar techniques such as sentiment analysis, automated summarization, and machine translation have allowed for more detailed and granular examination of historical source collections, enabling researchers to more accurately pinpoint specific sources most pertinent to their interests. However, each of these extraction techniques usually requires specialized models tailored to each task. LLMs, in contrast, offers the possibility of a all-in-one tool for performing various forms of data extraction in a single inference.

To demonstrate, in the following code block a LLM extracts metadata from the earlier oral history transcription and the OCR correction through use of a another [prompt](https://raw.githubusercontent.com/Dr-Hutchinson/jdh_submission/refs/heads/main/media/prompts/metadata_extraction_prompt.txt) containing examples for the LLM to emulate. For each source the LLM will offer a brief summary, perform sentiment analysis, and extract keywords for NER classifications in both English and German. This data is then structured into a machine-usable format as a JSON object.
<!-- #endregion -->

```python citation-manager={"citations": {"7amub": [{"id": "27937/LC63DETW", "source": "zotero"}], "kwh7q": [{"id": "27937/Z44J4BKC", "source": "zotero"}], "l0c94": [{"id": "27937/5ED45HQE", "source": "zotero"}], "mfahj": [{"id": "27937/LC63DETW", "source": "zotero"}], "z20ld": [{"id": "27937/5ED45HQE", "source": "zotero"}]}} jupyter={"outputs_hidden": false}
import requests
from openai import OpenAI
from rich.console import Console

# Initialize the console for rich output
console = Console()

# Function to fetch text content from a URL
def fetch_text_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful
    return response.text.strip()

# URL for the metadata extraction prompt
file_urls = {
    "metadata_extraction_prompt": "https://raw.githubusercontent.com/Dr-Hutchinson/jdh_submission/refs/heads/main/media/prompts/metadata_extraction_prompt.txt"
}

# Load the metadata extraction prompt from the URL
metadata_extraction_prompt = fetch_text_from_url(file_urls["metadata_extraction_prompt"])

# Initialize the OpenAI client
client = OpenAI()

# Function to query the LLM for metadata extraction with rich-formatted output
def extract_metadata(source_text, source_name):
    query = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": metadata_extraction_prompt + "\n" + source_text}
        ]
    )
    output = query.choices[0].message.content
    
    # Format the metadata output for display
    output_text = (
        f"[bold]Metadata Extraction for {source_name}[/bold]\n\n"
        f"{output}"
    )
    console.print(output_text)

# Whisper transcript and OCR corrected output from earlier code
sources = [
    {"text": whisper_transcript, "name": "Oral History Transcript"},
    {"text": gpt4_corrected_output_1, "name": "OCR Correction"}
]

# Run the extraction for each source
for source in sources:
    extract_metadata(source["text"], source["name"])

```

<!-- #region citation-manager={"citations": {"9k5i6": [{"id": "27937/CDKJEW4Z", "source": "zotero"}], "nkv58": [{"id": "27937/Z44J4BKC", "source": "zotero"}], "v3avs": [{"id": "27937/5ED45HQE", "source": "zotero"}], "wv7ar": [{"id": "27937/LC63DETW", "source": "zotero"}]}} jupyter={"outputs_hidden": false} -->
While neither the Whisper transcript nor the OCR correction were error-free, these errors did not prevent the LLM from successfully extracting relevant data in a structured JSON format. Further refinement of the prompt and incorporating more task-specific examples could improve the consistency and fidelity of such outputs. Additionally, fine-tuning LLM models for particular NER tasks and specific datasets can reduce misclassifications and the extraction of irrelevant entities. Researchers in various fields are already applying these techniques to extract data from multilingual historic text collections (<cite id="v3avs"><a href="#zotero%7C27937%2F5ED45HQE">González-Gallardo et al., “Leveraging Open Large Language Models for Historical Named Entity Recognition.”</a></cite>), as well in other fields for sources as varied as scientific papers (<cite id="wv7ar"><a href="#zotero%7C27937%2FLC63DETW">Dagdelen et al., “Structured Information Extraction from Scientific Text with Large Language Models.”</a></cite>), and electronic health records. (<cite id="nkv58"><a href="#zotero%7C27937%2FZ44J4BKC">Hu et al., “Improving Large Language Models for Clinical Named Entity Recognition via Prompt Engineering.”</a></cite>)

The implications of using LLMs for data extraction are significant. As Lauren Tilton argues: “As artificial intelligence generates data about data, guided by schemas, ontologies, and ways of seeing built into algorithms that will guide our search and aggregation, attention to the metadata and rethinking how artificial intelligence generates it will likely become a key part of historical research." (<cite id="9k5i6"><a href="#zotero%7C27937%2FCDKJEW4Z">Tilton, “Relating to Historical Sources.”</a></cite>) By streamlining tasks such as transcription, text correction, and metadata generation, LLMs can significantly reduce the workload of creating “tidy datasets”, enabling historians to focus on interpreting data instead of cleaning it. To be sure, this approach risks encoding hallucinations into the foundations of a data collection, and as emphasized in the previous case studies, human review remains essential. But with proper refinement, LLM-driven metadata extraction can prove a useful method for engaging with archival collections in new ways. Indeed, scholars are increasingly leveraging such data within larger computational systems to power new forms of scholarship using generative AI.
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} -->
## Case Study: Retrieval Augmented Generation (RAG) and Exploring Historic Text Collections
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"cwwxs": [{"id": "27937/P2KVKTMZ", "source": "zotero"}], "j022d": [{"id": "27937/7D6BEHLB", "source": "zotero"}], "k7j4i": [{"id": "27937/ECQ4J8E9", "source": "zotero"}], "usvx1": [{"id": "27937/HGR9QB96", "source": "zotero"}]}} jupyter={"outputs_hidden": false} -->
While LLMs offer a broad range of capabilities for data cleanup and extraction, their tendency to hallucinate remains a significant challenge for its use in historical research and analysis. However, recent advances in pairing generative AI models with other computational tools have helped ground LLM responses in greater factual accuracy. These techniques also enable LLMs to consult large text collections, search the Internet, and utilize external tools to solve problems in unfamiliar knowledge domains.

Retrieval augmented generation (RAG) has emerged as a leading approach in this space.  (<cite id="k7j4i"><a href="#zotero%7C27937%2FECQ4J8E9">Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.”</a></cite>) In this method a data collection is indexed and prepared for a retriever, which functions as a specialized search engine. When a user query submits a query, the retriever searches the data for the most relevant matches. While RAG retrievers typically feature keyword matching techniques found in search engines, they also use machine learning tools like text embeddings to provide results that are both semantically and contextually similar to the query. 
(<cite id="j022d"><a href="#zotero%7C27937%2F7D6BEHLB">Blankenship, Connell, and Dombrowski, “Understanding and Creating Word Embeddings.”</a></cite>) The highest scoring results are then passed to the LLM, which then generates a response grounded in the search results instead of hallucinated speculation. 

This approach is particularly useful for knowledge areas outside a LLM’s training data, like archival sources. An example of a RAG system in action is *Nicolay: Exploring the Speeches of Abraham Lincoln with AI* (<cite id="cwwxs"><a href="#zotero%7C27937%2FP2KVKTMZ">Hutchinson, “Nicolay: Exploring the Speeches of Abraham Lincoln with AI.”</a></cite>), a digital history project that uses RAG to analyze a collection of Lincoln speeches from the University of Virginia’s Miller Center. (<cite id="usvx1"><a href="#zotero%7C27937%2FHGR9QB96">“Presidential Speeches | Miller Center.”</a></cite>) While LLMs are well-versed in Lincoln’s life and writings, even the best generative AI models fall short on more narrow queries. For example, here is GPT’4’s response to the query: “What did Abraham Lincoln think of Japan?”
<!-- #endregion -->

```python jupyter={"outputs_hidden": false}
from openai import OpenAI
from rich.console import Console

# Initialize the console for rich output
console = Console()

# Initialize the OpenAI client
client = OpenAI()

# Create the query for the LLM
query = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "What did Abraham Lincoln think of Japan?"}
    ]
)

# Extract the output from the response
output = query.choices[0].message.content

# Format the output for display with rich
output_text = (
    "[bold cyan]GPT-4's Response:[/bold cyan]\n\n"
    f"{output}\n\n"
    "[bold cyan]End of GPT-4's Response[/bold cyan]"
)

# Display the formatted output
console.print(output_text, width=console.size.width)
```

<!-- #region jupyter={"outputs_hidden": false} -->
Here GPT-4 responds with general historical context to the question, which is valuable, but offers no specific insight into Lincoln’s own views, largely due to the lack of direct evidence in its training data. A LLM informed by RAG offers a very different response. Below is an output from *Nicolay*, which you can replicate via [this site.](https://nicolay-honestabes-info.streamlit.app/)
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} -->
**Begin Nicolay’s Response**

>Abraham Lincoln regarded Japan as an important partner in international relations, as evidenced by his references in annual messages to Congress. In his Second Annual Message, he noted, ‘Very favorable relations also continue to be maintained with Turkey, Morocco, China and Japan.’ (Second Annual Message, December 1, 1862) This statement reflects a period of stable and positive relations between the U.S. and Japan. However, Lincoln was also aware of the challenges in these relations. In his Third Annual Message, he expressed concern about the internal political struggles in Japan, stating, ‘In common with other western powers, our relations with Japan have been brought into serious jeopardy, through the perverse opposition of the hereditary aristocracy of the empire, to the enlightened and liberal policy of the Tycoon designed to bring the country into the society of nations.’ (Third Annual Message, December 8, 1863) This indicates his understanding of the impact of Japan’s internal politics on bilateral relations. Furthermore, in his Fourth Annual Message, Lincoln acknowledged the complexities of Japan’s political structure and the progress in treaty compliance, noting, ‘Owing to the peculiar situation of Japan, and the anomalous form of its government, the action of that empire in performing treaty stipulations is inconstant and capricious. Nevertheless, good progress has been effected by the western powers, moving with enlightened concert.’ (Fourth Annual Message, December 6, 1864) These statements collectively demonstrate Lincoln’s recognition of Japan’s significance in U.S. foreign policy and his efforts to navigate the complexities of this relationship during his presidency.
>
>References:
>
>Third Annual Message, December 8, 1863
>
>Fourth Annual Message, December 6, 1864
>
>Second Annual Message, December 1, 1862

**End Nicolay’s Response**
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"7ukai": [{"id": "27937/38C5LZCI", "source": "zotero"}], "fm8ii": [{"id": "27937/GP3PUHUJ", "source": "zotero"}], "hq9so": [{"id": "27937/MYBFNHF8", "source": "zotero"}], "q3vrr": [{"id": "27937/RJTNQXZP", "source": "zotero"}]}} jupyter={"outputs_hidden": false} -->
In less than a minute Nicolay searched through a corpus of over three hundred pages of text, identified the most pertinent sections, and returned an informed response containing direct quotes and accurate citations. This highlights how RAG systems can ground AI responses in factual detail while offering a chain of evidence and LLM ‘reasoning’ that historians can verify. RAG thus offers scholars a new way to directly query historical source collections, allowing for dialogue with historical data via a LLM intermediary.

Despite these advancements, RAG mitigates but does not eliminate the problem of hallucinations and errors. While increasingly sophisticated approaches are enhancing AI accuracy, LLMs can still misinterpret results in unexpected ways. Early RAG frameworks powering Google’s “AI Overviews” tool, for instance, famously returned nonsensical and sometimes dangerous responses to user queries, such as affirming the nutritional value of rocks and advising the use of glue to adhere cheese to pizzas. (<cite id="7ukai"><a href="#zotero%7C27937%2F38C5LZCI">McMahon and Kleinman, “Google AI Search Tells Users to Glue Pizza and Eat Rocks.”</a></cite>) RAG systems can serve as useful research tools, but the responses they generate should be treated as starting points for deeper analysis, not as final conclusions.

Scholars are already experimenting with the potential of these systems. Developers from the Library Innovation Lab at Harvard Law School Library have released a framework for enabling RAG search over web archive collections, enabling chatbots to return relevant data based on user queries. (<cite id="hq9so"><a href="#zotero%7C27937%2FMYBFNHF8">Cargnelutti, Mukk, and Stanton, “WARC-GPT.”</a></cite>) Among the most impressive RAG applications developed to date is [STORM](https://storm.genie.stanford.edu/) from Stanford University’s Virtual Open Assistant Lab. This advanced RAG application utilizes a complex framework of LLM “experts” to search the web and construct Wikipedia-style articles on demand. (<cite id="q3vrr"><a href="#zotero%7C27937%2FRJTNQXZP">Shao et al., “Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models.”</a></cite>) While Nicolay produced a paragraph-length response to the Lincoln query above based on a limited corpus, STORM produced a [multi-page overview](https://github.com/Dr-Hutchinson/jdh_submission/blob/main/media/STORM%20Essay%20-%20Abraham%20Lincoln's%20views%20on%20Japan..pdf) supported by citations from sources across the Internet (although, notably, not from Lincoln’s speeches).

While such forms of programmatic essay writing are still in its infancy, the increasing capacities of LLMs to aid automated research has significant implications. While these RAG systems have exciting potential, they also raise important questions about the evolving place of these technologies for digital history. As Benjamin Schmidt notes, such AI-powered systems “don’t only give us new haystacks to search in; they also change the types of needles people will find.” (<cite id="fm8ii"><a href="#zotero%7C27937%2FGP3PUHUJ">Schmidt, “Representation Learning.”</a></cite>) While the potential of generative AI as a historical research tool appears significant, their existing limits makes clear that human expertise will continue to be central in guiding and contextualizing these new technologies. Such expertise will be increasingly important, especially as generative AI leads to novel forms of historical exploration and representation.
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} -->
# The Past as Latent Space: Exploring New Frontiers in Digital History
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"39enb": [{"id": "27937/I363EKXY", "source": "zotero"}], "63boa": [{"id": "27937/XBWIZZJJ", "source": "zotero"}], "9bzc9": [{"id": "27937/P3ZKA48D", "source": "zotero"}], "cczf8": [{"id": "27937/XQYUJV5F", "source": "zotero"}], "dqw8n": [{"id": "27937/CXURVMLQ", "source": "zotero"}], "g2475": [{"id": "27937/82YIALT5", "source": "zotero"}], "gdfmp": [{"id": "27937/VHJBTADE", "source": "zotero"}], "l3aq9": [{"id": "27937/JUTZSXVB", "source": "zotero"}], "pdy8m": [{"id": "27937/9CA225ZV", "source": "zotero"}], "rnhet": [{"id": "27937/KPPP2BAQ", "source": "zotero"}], "sbrz8": [{"id": "27937/USYR9HC8", "source": "zotero"}], "uqhgl": [{"id": "27937/BN44JR8V", "source": "zotero"}]}} jupyter={"outputs_hidden": false} -->
These case studies demonstrate how LLMs can be prompted to assist digital historians in familiar tasks - transcribing and correcting text, extracting data, even finding connections within archival collections. Yet, it’s often in unexpected domains where LLMs reveal their most intriguing and ethically complex implications. As Ted Underwood argues, these technologies enable us to explore the “latent spaces” of culture - the underlying patterns, assumptions, and potential meanings embedded within the vast datasets that shape our understanding of the past. (<cite id="uqhgl"><a href="#zotero%7C27937%2FBN44JR8V">Underwood, “Mapping the Latent Spaces of Culture.”</a></cite>) As historians increasingly integrate LLMs into their work, they are discovering that generative AI is not simply streamlining old practices; it is also empowering entirely new forms of historical representation, for both good and ill.

One area where this technology has been felt is in the classroom. For many instructors, their first encounter with generative AI involves suspicion: are student submissions genuine or AI-produced? This concern, compounded by the frustrating unreliability of current LLM-detection tools, has led many to “AI-proof” their assignments and even some to declare that “the essay is dead.” (<cite id="9bzc9"><a href="#zotero%7C27937%2FP3ZKA48D">Marche, “Will ChatGPT Kill the Student Essay? - The Atlantic.”</a></cite>) Collective responses to the impact of this technology are emerging from organizations like the MLA-CCCC’s Working Group on Writing and AI. (<cite id="sbrz8"><a href="#zotero%7C27937%2FUSYR9HC8">MLA-CCCC Joint Task Force on Writing and AI, “Using the Student Guide to AI Literacy – MLA-CCCC Joint Task Force on Writing and AI.”</a></cite>) Historians are joining these efforts in crafting disciplinary responses to generative AI. (<cite id="cczf8"><a href="#zotero%7C27937%2FXQYUJV5F">Meadows and Sternfeld, “Artificial Intelligence and the Practice of History.”</a></cite>)

Emerging from such discussions are innovative pedagogies that apply and critique generative AI. (<cite id="dqw8n"><a href="#zotero%7C27937%2FCXURVMLQ">Vee, Laquintano, and Schnitzler, “TextGenEd Exhibit.”</a></cite>) One approach shared by Benjamin Breen merits particular attention. Breen has designed prompts to transform LLMs into narrative game engines for interactive historical simulations, akin to “choose-your-own-adventure” games. Students navigate these simulated worlds and converse with AI-controlled characters. Students then annotate their ‘dialogues’ against assigned historical sources, helping contrast the model’s capacities for accuracy against its tendencies towards hallucination and bias. As Breen noted in his first interaction of this assignment, “Student engagement in the spring quarter, when I began these trials, was unlike anything I’ve seen….[the] assignments worked wonders in terms of sparking enthusiasm among previously disengaged students." (<cite id="l3aq9"><a href="#zotero%7C27937%2FJUTZSXVB">Breen, “Simulating History with ChatGPT.”</a></cite>) Breen’s creative approach reflects real opportunities for teaching with generative AI in inventive and effective ways.

While creative pedagogies offer exciting possibilities for using and critiquing LLMs in the classroom, the ethical dilemmas posed by these technologies extend far beyond educational contexts, particularly when it comes to representing historical figures. One notable niche among generative AI services are programmable chatbot ‘personalities,’ such as prominent personalities from the past. Following the release of ChatGPT in November 2022, dialogues with AI-impersonations of iconic and controversial figures were widely shared over social media. App developers quickly tapped into what they perceived as an engaging approach to representing the past. However, these apps did little to address the problems of LLM “hallucinations,” nor did they take into account the impact of training biases in LLM “knowledge.” Users soon reported disturbing conversations with both humanity’s greatest luminaries and its greatest villains. (<cite id="rnhet"><a href="#zotero%7C27937%2FKPPP2BAQ">“Chatbot That Lets You Talk to Jesus and Hitler Is Latest AI Controversy.”</a></cite>) The ability of these apps to “bring history to life” soon gave way to an appreciation that perhaps some parts of the past are better off dead.

Such concerns are particularly important as such resurrection can now occur at scale. Researchers are exploring how LLMs can be prompted to emulate human behaviors and serve as proxies for human research subjects in fields as varied as psychology (<cite id="g2475"><a href="#zotero%7C27937%2F82YIALT5">Cui, Li, and Zhou, “Can AI Replace Human Subjects?”</a></cite>), behavioral economics (<cite id="39enb"><a href="#zotero%7C27937%2FI363EKXY">Xie et al., “Can Large Language Model Agents Simulate Human Trust Behaviors?”</a></cite>), and public health. (<cite id="pdy8m"><a href="#zotero%7C27937%2F9CA225ZV">Stade et al., “Large Language Models Could Change the Future of Behavioral Healthcare.”</a></cite>) A compelling study by Stanford researchers in 2023 demonstrates how such emulation could be scaled to simulate entire communities. In their paper “Generative Agents: Interactive Simulacra of Human Behavior,” researchers created a framework to generate a small town populated with residents possessing programmed behaviors and beliefs. LLMs then guided each individual resident as they completed daily tasks and encountered the world around them. As time progressed the inhabitants interacted with each other, reflected on their encounters, and updated their “memories” based on their experiences. Researchers observing the LLM-powered “agents” noted the emergence of spontaneous actions and autonomous planning, leading to unexpected but plausible collective behaviors. (<cite id="63boa"><a href="#zotero%7C27937%2FXBWIZZJJ">Park et al., “Generative Agents.”</a></cite>) Such paradigms offer potential methods to model historical behaviors at scale and perhaps enable simulation as a new empirical approach. However, historians of earlier forms of computational modeling of human behavior would be quick to remind us of the abuses and errors from previous generations of “artificial intelligence” and warn us of the risks of such approaches. (<cite id="gdfmp"><a href="#zotero%7C27937%2FVHJBTADE">Lepore, <i>If Then</i>.</a></cite>)

Indeed, as LLMs increasingly become part of our digital lives, historians should move beyond simply using these technologies to using their scholarly expertise to shape the debates concerning their development and implementation. We have a distinctive perspective to share. LLMs are not just tools; they are also historical sources. And like every source they are flawed, anchored in their time and place, and influenced by a particular and often distorted view of the world. But within these limitations lies real potential to apply the vast historical data on which they are trained. Historians are already using LLMs to explore the “latent space” of the past, and offering pointed critiques on how to contextualize the significance of this technology. By approaching LLMs critically, ethically, and collaboratively, digital historians are contributing to Roy Rosenzweig’s vision of navigating the “unheard-of historical abundance” of the digital age in a manner that ensures that these technologies deepen our understanding of the past, and not distort it.
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} -->
I am grateful to Abraham Gibson for extending an invitation to present the preliminary research findings of this article with the Digital History Working Group in May 2022, organized by the Consortium For History of Science, Technology, and Medicine. I would also like to express my appreciation to my colleagues William Mattingly, Patrick Wadden, and Ian Crowe for their insightful commentary on the article, and to the editorial staff and reviewers for the Journal of Digital History. This article was facilitated by a sabbatical semester generously granted by the Office of Academic Affairs at Belmont Abbey College. My thanks to Provost Travis Feezell and Vice Provost David Williams for their support of this project.
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["hidden"] -->
# Bibliography
<!-- #endregion -->

<!-- #region jupyter={"outputs_hidden": false} slideshow={"slide_type": ""} tags=["hidden"] -->
<!-- BIBLIOGRAPHY START -->
<div class="csl-bib-body">
  <div class="csl-entry"><i id="zotero|27937/58X69RSW"></i>Abdin, Marah, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, et al. “Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone.” arXiv, August 30, 2024. <a href="http://arxiv.org/abs/2404.14219">http://arxiv.org/abs/2404.14219</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/5YDNQS4V"></i>Barton, Nicol Turner Lee, Paul Resnick, and Genie. “Algorithmic Bias Detection and Mitigation: Best Practices and Policies to Reduce Consumer Harms.” <i>Brookings</i> (blog), May 22, 2019. <a href="https://www.brookings.edu/research/algorithmic-bias-detection-and-mitigation-best-practices-and-policies-to-reduce-consumer-harms/">https://www.brookings.edu/research/algorithmic-bias-detection-and-mitigation-best-practices-and-policies-to-reduce-consumer-harms/</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/MVDFMR8K"></i>Bender, Emily, Timnit Gebru, Angelina McMillan-Major, and Margaret Mitchell. “On the Dangers of Stochastic Parrots | Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency.” Accessed March 27, 2023. <a href="https://dl.acm.org/doi/10.1145/3442188.3445922">https://dl.acm.org/doi/10.1145/3442188.3445922</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/TPGPSRAI"></i>Bender, Emily M. “On NYT Magazine on AI: Resist the Urge to Be Impressed.” <i>Medium</i> (blog), May 2, 2022. <a href="https://medium.com/@emilymenonbender/on-nyt-magazine-on-ai-resist-the-urge-to-be-impressed-3d92fd9a0edd">https://medium.com/@emilymenonbender/on-nyt-magazine-on-ai-resist-the-urge-to-be-impressed-3d92fd9a0edd</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/X4D92B7V"></i>Benjamin, Ruha. <i>Race After Technology: Abolitionist Tools for the New Jim Code</i>. 1st edition. Medford, MA: Polity, 2019.</div>
  <div class="csl-entry"><i id="zotero|27937/7D6BEHLB"></i>Blankenship, Avery, Sarah Connell, and Quinn Dombrowski. “Understanding and Creating Word Embeddings.” <i>Programming Historian</i>, January 31, 2024. <a href="https://programminghistorian.org/en/lessons/understanding-creating-word-embeddings">https://programminghistorian.org/en/lessons/understanding-creating-word-embeddings</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/F3XT4XAQ"></i>Bommasani, Rishi, Drew A. Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein, et al. “On the Opportunities and Risks of Foundation Models.” arXiv, July 12, 2022. <a href="http://arxiv.org/abs/2108.07258">http://arxiv.org/abs/2108.07258</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/JUTZSXVB"></i>Breen, Benjamin. “Simulating History with ChatGPT.” Substack newsletter. <i>Res Obscura</i> (blog), September 12, 2023. <a href="https://resobscura.substack.com/p/simulating-history-with-chatgpt">https://resobscura.substack.com/p/simulating-history-with-chatgpt</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/KNEK45E4"></i>Brown, Tom B., Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, et al. “Language Models Are Few-Shot Learners.” arXiv, July 22, 2020. <a href="http://arxiv.org/abs/2005.14165">http://arxiv.org/abs/2005.14165</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/MYBFNHF8"></i>Cargnelutti, Matteo, Kristi Mukk, and Clare Stanton. “WARC-GPT: An Open-Source Tool for Exploring Web Archives Using AI | Library Innovation Lab.” Library Innovation Lab Blog, Harvard Law Library, February 12, 2024. <a href="https://lil.law.harvard.edu/blog/2024/02/12/warc-gpt-an-open-source-tool-for-exploring-web-archives-with-ai/">https://lil.law.harvard.edu/blog/2024/02/12/warc-gpt-an-open-source-tool-for-exploring-web-archives-with-ai/</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/HIPL38QS"></i>Chastang, Pierre, Sergio Torres Aguilar, and Xavier Tannier. “A Named Entity Recognition Model for Medieval Latin Charters.” <i>Digital Humanities Quarterly</i> 15, no. 4 (2021). <a href="https://www.digitalhumanities.org/dhq/vol/15/4/000574/000574.html">https://www.digitalhumanities.org/dhq/vol/15/4/000574/000574.html</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/KPPP2BAQ"></i>NBC News. “Chatbot That Lets You Talk to Jesus and Hitler Is Latest AI Controversy,” January 20, 2023. <a href="https://www.nbcnews.com/tech/tech-news/chatgpt-gpt-chat-bot-ai-hitler-historical-figures-open-rcna66531">https://www.nbcnews.com/tech/tech-news/chatgpt-gpt-chat-bot-ai-hitler-historical-figures-open-rcna66531</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/FMW5DCWM"></i>Chowdhery, Aakanksha, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, et al. “PaLM: Scaling Language Modeling with Pathways.” arXiv, October 5, 2022. <a href="http://arxiv.org/abs/2204.02311">http://arxiv.org/abs/2204.02311</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/TGPDB8WX"></i>Christiano, Paul, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. “Deep Reinforcement Learning from Human Preferences.” arXiv, February 17, 2023. <a href="https://doi.org/10.48550/arXiv.1706.03741">https://doi.org/10.48550/arXiv.1706.03741</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/YVTAGDKZ"></i>Crawford, Kate. <i>Atlas of AI: Power, Politics, and the Planetary Costs of Artificial Intelligence</i>. New Haven: Yale University Press, 2021.</div>
  <div class="csl-entry"><i id="zotero|27937/5GTQD5W9"></i>Crawford, Kate, and Trevor Paglen. “Excavating AI.” Excavating AI. Accessed March 28, 2023. <a href="https://excavating.ai">https://excavating.ai</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/82YIALT5"></i>Cui, Ziyan, Ning Li, and Huaikang Zhou. “Can AI Replace Human Subjects? A Large-Scale Replication of Psychological Experiments with LLMs.” arXiv, September 4, 2024. <a href="https://doi.org/10.48550/arXiv.2409.00128">https://doi.org/10.48550/arXiv.2409.00128</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/LC63DETW"></i>Dagdelen, John, Alexander Dunn, Sanghoon Lee, Nicholas Walker, Andrew S. Rosen, Gerbrand Ceder, Kristin A. Persson, and Anubhav Jain. “Structured Information Extraction from Scientific Text with Large Language Models.” <i>Nature Communications</i> 15, no. 1 (February 15, 2024): 1418. <a href="https://doi.org/10.1038/s41467-024-45563-x">https://doi.org/10.1038/s41467-024-45563-x</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/VXGSAGTI"></i>Dasu, Tamraparni, and Theodore Johnson. <i>Exploratory Data Mining and Data Cleaning</i>. 1st edition. New York: Wiley-Interscience, 2003.</div>
  <div class="csl-entry"><i id="zotero|27937/GHGWH4HI"></i>Dzieza, Josh. “What AI Can Do for Historians.” The Verge, February 15, 2024. <a href="https://www.theverge.com/24068716/ai-historians-academia-llm-chatgpt">https://www.theverge.com/24068716/ai-historians-academia-llm-chatgpt</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/QD3X7XMD"></i>Eloundou, Tyna, Sam Manning, Pamela Mishkin, and Daniel Rock. “GPTs Are GPTs: An Early Look at the Labor Market Impact Potential of Large Language Models.” arXiv, March 23, 2023. <a href="http://arxiv.org/abs/2303.10130">http://arxiv.org/abs/2303.10130</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/S3ADX5DD"></i>Gebru, Timnit. “Race and Gender.” In <i>The Oxford Handbook of Ethics of AI</i>, edited by Markus D. Dubber, Frank Pasquale, and Sunit Das, 0. Oxford University Press, 2020. <a href="https://doi.org/10.1093/oxfordhb/9780190067397.013.16">https://doi.org/10.1093/oxfordhb/9780190067397.013.16</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/MHRIEHH8"></i>Gehman, Samuel, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A. Smith. “RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models.” arXiv, September 25, 2020. <a href="http://arxiv.org/abs/2009.11462">http://arxiv.org/abs/2009.11462</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/5ED45HQE"></i>González-Gallardo, Carlos-Emiliano, Tran Thi Hong Hanh, Ahmed Hamdi, and Antoine Doucet. “Leveraging Open Large Language Models for Historical Named Entity Recognition,” 2024. <a href="https://univ-rochelle.hal.science/hal-04662000">https://univ-rochelle.hal.science/hal-04662000</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/L2ILKERU"></i>Graham, Shawn, Ian Milligan, and Scott Weingart. <i>Exploring Big Historical Data: The Historian’s Macroscope</i>. Reprint edition. London: Icp, 2015.</div>
  <div class="csl-entry"><i id="zotero|27937/A834FRJL"></i>Hendrycks, Dan. <i>Measuring Massive Multitask Language Understanding</i>, 2023. <a href="https://github.com/hendrycks/test">https://github.com/hendrycks/test</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/ZS9JDNGD"></i>Hendrycks, Dan, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. “Measuring Massive Multitask Language Understanding.” arXiv, January 12, 2021. <a href="http://arxiv.org/abs/2009.03300">http://arxiv.org/abs/2009.03300</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/G5ESJ8NI"></i>Hoffmann, Jordan, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, et al. “Training Compute-Optimal Large Language Models.” arXiv, March 29, 2022. <a href="http://arxiv.org/abs/2203.15556">http://arxiv.org/abs/2203.15556</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/Z44J4BKC"></i>Hu, Yan, Qingyu Chen, Jingcheng Du, Xueqing Peng, Vipina Kuttichi Keloth, Xu Zuo, Yujia Zhou, et al. “Improving Large Language Models for Clinical Named Entity Recognition via Prompt Engineering.” <i>Journal of the American Medical Informatics Association</i> 31, no. 9 (September 1, 2024): 1812–20. <a href="https://doi.org/10.1093/jamia/ocad259">https://doi.org/10.1093/jamia/ocad259</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/P2KVKTMZ"></i>Hutchinson, Daniel. “Nicolay: Exploring the Speeches of Abraham Lincoln with AI.” Honest Abe’s Information Emporium., 2023. <a href="https://nicolay-honestabes-info.streamlit.app/">https://nicolay-honestabes-info.streamlit.app/</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/5AL5LZ2K"></i>———. What Do AIs Know About History? A Digital History Experiment, 2022. <a href="https://dr-hutchinson-what-do-ais-know-about-history-app-i3l5jo.streamlit.app/">https://dr-hutchinson-what-do-ais-know-about-history-app-i3l5jo.streamlit.app/</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/9GQG6VFM"></i>Ji, Ziwei, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, et al. “Survey of Hallucination in Natural Language Generation.” <i>ACM Computing Surveys</i> 55, no. 12 (December 31, 2023): 1–38. <a href="https://doi.org/10.1145/3571730">https://doi.org/10.1145/3571730</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/H9BUWE28"></i>Kaplan, Jared, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. “Scaling Laws for Neural Language Models.” arXiv, January 22, 2020. <a href="https://doi.org/10.48550/arXiv.2001.08361">https://doi.org/10.48550/arXiv.2001.08361</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/BVBZMR66"></i>Katz, Daniel Martin. “GPT Takes the Bar Exam.” arXiv, December 29, 2022. <a href="https://doi.org/10.48550/arXiv.2212.14402">https://doi.org/10.48550/arXiv.2212.14402</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/4ITT4MQK"></i>Koenecke, Allison, Anna Seo Gyeong Choi, Katelyn X. Mei, Hilke Schellmann, and Mona Sloane. “Careless Whisper: Speech-to-Text Hallucination Harms.” arXiv, May 3, 2024. <a href="https://doi.org/10.48550/arXiv.2402.08021">https://doi.org/10.48550/arXiv.2402.08021</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/JV9GGCQA"></i>Langlais, Pierre-Carl. “Post-OCR-Correction: 1 Billion Words Dataset of Automated OCR Correction by LLM.” Accessed October 14, 2024. <a href="https://huggingface.co/blog/Pclanglais/post-ocr-correction">https://huggingface.co/blog/Pclanglais/post-ocr-correction</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/KKDPZJYW"></i>Lehečka, Jan, Jan Švec, Josef V. Psutka, and Pavel Ircing. “Transformer-Based Speech Recognition Models for Oral History Archives in English, German, and Czech.” In <i>INTERSPEECH 2023</i>, 201–5. ISCA, 2023. <a href="https://doi.org/10.21437/Interspeech.2023-872">https://doi.org/10.21437/Interspeech.2023-872</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/VHJBTADE"></i>Lepore, Jill. <i>If Then: How the Simulmatics Corporation Invented the Future</i>. New York, NY: Liveright Publishing Corporation, a division of W.W. Norton &#38; Company, 2021.</div>
  <div class="csl-entry"><i id="zotero|27937/ECQ4J8E9"></i>Lewis, Patrick, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, et al. “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.” arXiv, April 12, 2021. <a href="https://doi.org/10.48550/arXiv.2005.11401">https://doi.org/10.48550/arXiv.2005.11401</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/GSIXPJ7P"></i>Mai, Yifan, and Percy Liang. “Massive Multitask Language Understanding (MMLU) on HELM.” Blog. Center for Research on Foundation Models, Stanford University, May 1, 2024. <a href="https://crfm.stanford.edu/2024/05/01/helm-mmlu.html">https://crfm.stanford.edu/2024/05/01/helm-mmlu.html</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/P3ZKA48D"></i>Marche, Stephen. “Will ChatGPT Kill the Student Essay? - The Atlantic.” The Atlantic, December 6, 2022. <a href="https://www.theatlantic.com/technology/archive/2022/12/chatgpt-ai-writing-college-student-essays/672371/">https://www.theatlantic.com/technology/archive/2022/12/chatgpt-ai-writing-college-student-essays/672371/</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/78DL3V96"></i>Marshall, Lindsay. “The Strange World of AP U.S. History.” CONTINGENT, October 20, 2020. <a href="https://contingentmagazine.org/2020/10/20/apush/">https://contingentmagazine.org/2020/10/20/apush/</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/EZNK3CE3"></i>McCorduck, Pamela. <i>Machines Who Think a Personal Inquiry into the History and Prospects of Artificial Intelligence</i>. 25th anniversary update. Natick, Mass.: A.K. Peters, 2004. <a href="http://site.ebrary.com/id/10158052">http://site.ebrary.com/id/10158052</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/38C5LZCI"></i>McMahon, Liv, and Zoe Kleinman. “Google AI Search Tells Users to Glue Pizza and Eat Rocks.” BBC News. Accessed October 16, 2024. <a href="https://www.bbc.com/news/articles/cd11gzejgz4o">https://www.bbc.com/news/articles/cd11gzejgz4o</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/XQYUJV5F"></i>Meadows, R. Darrell, and Joshua Sternfeld. “Artificial Intelligence and the Practice of History.” <i>The American Historical Review</i> 128, no. 3 (September 26, 2023): 1345–49. <a href="https://doi.org/10.1093/ahr/rhad362">https://doi.org/10.1093/ahr/rhad362</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/ZJW9AI49"></i>Milligan, Ian. “Illusionary Order: Online Databases, Optical Character Recognition, and Canadian History, 1997–2010.” <i>Canadian Historical Review</i> 94, no. 4 (December 2013): 540–69. <a href="https://doi.org/10.3138/chr.694">https://doi.org/10.3138/chr.694</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/USYR9HC8"></i>MLA-CCCC Joint Task Force on Writing and AI. “Using the Student Guide to AI Literacy – MLA-CCCC Joint Task Force on Writing and AI.” <i>MLA-CCCC Joint Task Force on Writing and AI</i> (blog), October 8, 2004. <a href="https://aiandwriting.hcommons.org/2024/10/08/using-the-student-guide-to-ai-literacy/">https://aiandwriting.hcommons.org/2024/10/08/using-the-student-guide-to-ai-literacy/</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/TIAJYHF6"></i>Muehlberger, Guenter, Louise Seaward, Melissa Terras, Sofia Ares Oliveira, Vicente Bosch, Maximilian Bryan, Sebastian Colutto, et al. “Transforming Scholarship in the Archives through Handwritten Text Recognition: Transkribus as a Case Study.” <i>Journal of Documentation</i> 75, no. 5 (January 1, 2019): 954–76. <a href="https://doi.org/10.1108/JD-07-2018-0114">https://doi.org/10.1108/JD-07-2018-0114</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/68YHDUH6"></i>Nikolova, Ivelina, and Michael Levy. “Using Named Entity Recognition to Enhance Access to a Museum Catalog – Document Blog.” <i>The European Holocaust Research Infrastructure Document Blog</i> (blog), August 27, 2018. <a href="https://blog.ehri-project.eu/2018/08/27/named-entity-recognition/">https://blog.ehri-project.eu/2018/08/27/named-entity-recognition/</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/IEQ8GAVU"></i>Noble, Safiya Umoja. <i>Algorithms of Oppression: How Search Engines Reinforce Racism</i>. Illustrated edition. New York: NYU Press, 2018.</div>
  <div class="csl-entry"><i id="zotero|27937/VEDFUUBA"></i>Nori, Harsha, Nicholas King, Scott Mayer McKinney, Dean Carignan, and Eric Horvitz. “Capabilities of GPT-4 on Medical Challenge Problems.” arXiv, March 20, 2023. <a href="http://arxiv.org/abs/2303.13375">http://arxiv.org/abs/2303.13375</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/U534FF7L"></i>OpenAI. “GPT-4 Technical Report.” arXiv, March 27, 2023. <a href="http://arxiv.org/abs/2303.08774">http://arxiv.org/abs/2303.08774</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/XBWIZZJJ"></i>Park, Joon Sung, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. “Generative Agents: Interactive Simulacra of Human Behavior.” arXiv, August 6, 2023. <a href="http://arxiv.org/abs/2304.03442">http://arxiv.org/abs/2304.03442</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/ZXTQBIJU"></i>PleIAs. “PleIAs/Post-OCR-Correction · Datasets at Hugging Face.” PleIAs/Post-OCR-Correction · Datasets at Hugging Face, June 9, 2024. <a href="https://huggingface.co/datasets/PleIAs/Post-OCR-Correction">https://huggingface.co/datasets/PleIAs/Post-OCR-Correction</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/HGR9QB96"></i>“Presidential Speeches | Miller Center,” November 21, 2016. <a href="https://millercenter.org/the-presidency/presidential-speeches">https://millercenter.org/the-presidency/presidential-speeches</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/BD8996H7"></i>AP Exam Administration Data Archive. “Program Summary Report,” 2022. <a href="https://reports.collegeboard.org/media/pdf/program-summary-report-2022.pdf">https://reports.collegeboard.org/media/pdf/program-summary-report-2022.pdf</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/UYVGUT4C"></i>Radford, Alec, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, et al. “Learning Transferable Visual Models From Natural Language Supervision.” arXiv, February 26, 2021. <a href="http://arxiv.org/abs/2103.00020">http://arxiv.org/abs/2103.00020</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/7VHKCH3M"></i>Radford, Alec, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. “Robust Speech Recognition via Large-Scale Weak Supervision.” arXiv, December 6, 2022. <a href="http://arxiv.org/abs/2212.04356">http://arxiv.org/abs/2212.04356</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/YWJAQ4V8"></i>Ritchie, Donald A. <i>Doing Oral History: A Practical Guide</i>. 2nd ed. Cary: Oxford University Press, USA, 2003.</div>
  <div class="csl-entry"><i id="zotero|27937/MYFQUX4C"></i>Rochester Institute of Technology. “Artificial Intelligence Aids Cultural Heritage Researchers Documenting and Teaching Oral Histories.” Artificial intelligence aids cultural heritage researchers documenting and teaching oral histories. Accessed October 10, 2024. <a href="https://www.rit.edu/news/artificial-intelligence-aids-cultural-heritage-researchers-documenting-and-teaching-oral">https://www.rit.edu/news/artificial-intelligence-aids-cultural-heritage-researchers-documenting-and-teaching-oral</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/CJYNFHVI"></i>Rosenzweig, Roy. “Scarcity or Abundance? Preserving the Past in a Digital Era.” <i>The American Historical Review</i> 108, no. 3 (June 1, 2003): 735–62. <a href="https://doi.org/10.1086/ahr/108.3.735">https://doi.org/10.1086/ahr/108.3.735</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/5G5LJCLC"></i>Saravia, Elvis. <i>Prompt Engineering Guide</i>, 2022. <a href="https://github.com/dair-ai/Prompt-Engineering-Guide">https://github.com/dair-ai/Prompt-Engineering-Guide</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/GP3PUHUJ"></i>Schmidt, Benjamin. “Representation Learning.” <i>The American Historical Review</i> 128, no. 3 (September 26, 2023): 1350–53. <a href="https://doi.org/10.1093/ahr/rhad363">https://doi.org/10.1093/ahr/rhad363</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/I2BKP7MN"></i>Schultz, Emilien. “[Tutorial] Using Whisper to Transcribe Oral Interviews – CSS @ IPP,” February 12, 2024. <a href="https://www.css.cnrs.fr/using-whisper-to-transcribe-oral-interviews/">https://www.css.cnrs.fr/using-whisper-to-transcribe-oral-interviews/</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/RJTNQXZP"></i>Shao, Yijia, Yucheng Jiang, Theodore A. Kanell, Peter Xu, Omar Khattab, and Monica S. Lam. “Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models.” arXiv, April 8, 2024. <a href="https://doi.org/10.48550/arXiv.2402.14207">https://doi.org/10.48550/arXiv.2402.14207</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/UHZYQM3W"></i>Somers, James. “Whispers of A.I.’s Modular Future | The New Yorker,” February 1, 2023. <a href="https://www.newyorker.com/tech/annals-of-technology/whispers-of-ais-modular-future">https://www.newyorker.com/tech/annals-of-technology/whispers-of-ais-modular-future</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/9CA225ZV"></i>Stade, Elizabeth C., Shannon Wiltsey Stirman, Lyle H. Ungar, Cody L. Boland, H. Andrew Schwartz, David B. Yaden, João Sedoc, Robert J. DeRubeis, Robb Willer, and Johannes C. Eichstaedt. “Large Language Models Could Change the Future of Behavioral Healthcare: A Proposal for Responsible Development and Evaluation.” <i>Npj Mental Health Research</i> 3, no. 1 (April 2, 2024): 1–12. <a href="https://doi.org/10.1038/s44184-024-00056-z">https://doi.org/10.1038/s44184-024-00056-z</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/NYNDVYMM"></i>Strickland, Eliza. “OpenAI’s GPT-3 Speaks! (Kindly Disregard Toxic Language) - IEEE Spectrum.” OpenAI’s GPT-3 Speaks! (Kindly Disregard Toxic Language) - IEEE Spectrum, February 1, 2021. <a href="https://spectrum.ieee.org/open-ais-powerful-text-generating-tool-is-ready-for-business">https://spectrum.ieee.org/open-ais-powerful-text-generating-tool-is-ready-for-business</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/CDKJEW4Z"></i>Tilton, Lauren. “Relating to Historical Sources.” <i>The American Historical Review</i> 128, no. 3 (September 26, 2023): 1354–59. <a href="https://doi.org/10.1093/ahr/rhad365">https://doi.org/10.1093/ahr/rhad365</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/BN44JR8V"></i>Underwood, Ted. “Mapping the Latent Spaces of Culture,” October 20, 2021. <a href="https://hcommons.org/deposits/item/hc:41973/">https://hcommons.org/deposits/item/hc:41973/</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/9T2I7QLM"></i>Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. “Attention Is All You Need.” arXiv, August 2, 2023. <a href="http://arxiv.org/abs/1706.03762">http://arxiv.org/abs/1706.03762</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/3H7M3AJ8"></i>Vatsal, Shubham, and Harsh Dubey. “A Survey of Prompt Engineering Methods in Large Language Models for Different NLP Tasks.” arXiv, July 24, 2024. <a href="https://doi.org/10.48550/arXiv.2407.12994">https://doi.org/10.48550/arXiv.2407.12994</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/CXURVMLQ"></i>Vee, Annette, Tim Laquintano, and Carly Schnitzler. “TextGenEd Exhibit.” <i>The WAC Repository</i> 1, no. 1 (2023): 1–100. <a href="https://doi.org/10.37514/TWR-J.2023.1.1.02">https://doi.org/10.37514/TWR-J.2023.1.1.02</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/56EE9N63"></i>Wei, Jason, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, et al. “Emergent Abilities of Large Language Models.” arXiv, October 26, 2022. <a href="http://arxiv.org/abs/2206.07682">http://arxiv.org/abs/2206.07682</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/XEUKQDPE"></i>Wickham, Hadley. “Tidy Data.” <i>Journal of Statistical Software</i> 59 (September 12, 2014): 1–23. <a href="https://doi.org/10.18637/jss.v059.i10">https://doi.org/10.18637/jss.v059.i10</a>.</div>
  <div class="csl-entry"><i id="zotero|27937/I363EKXY"></i>Xie, Chengxing, Canyu Chen, Feiran Jia, Ziyu Ye, Kai Shu, Adel Bibi, Ziniu Hu, Philip Torr, Bernard Ghanem, and Guohao Li. “Can Large Language Model Agents Simulate Human Trust Behaviors?” arXiv, March 10, 2024. <a href="https://doi.org/10.48550/arXiv.2402.04559">https://doi.org/10.48550/arXiv.2402.04559</a>.</div>
</div>
<!-- BIBLIOGRAPHY END -->
<!-- #endregion -->
