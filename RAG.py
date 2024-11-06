import os
import glob
import json
from typing import Dict, List, Tuple, Union
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sentence_transformers import SentenceTransformer
import transformers
import torch
from PIL import Image
import faiss

# Utility functions (unchanged)
def read_mathpix_json(json_path: str) -> Dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_title_and_abstract(mathpix_data: Union[str, Dict]) -> Tuple[str, str]:
    if isinstance(mathpix_data, str):
        mathpix_data = json.loads(mathpix_data)
    
    title = ""
    abstract = ""
    abstract_start = False
    
    for page in mathpix_data['pages']:
        for line in page['lines']:
            text = line.get('text', '')
            if not title and '\\title' in text:
                title = text.replace('\\title{', '').replace('}', '').strip()
            if '\\begin{abstract}' in text:
                abstract_start = True
                text = text.replace('\\begin{abstract}', '').strip()
            if abstract_start:
                if '\\end{abstract}' in text:
                    abstract += text.split('\\end{abstract}')[0].strip()
                    return title, abstract.strip()
                abstract += text + "\n"

    return title, abstract.strip()

def get_sections(mathpix_json: Dict) -> OrderedDict:
    sections = OrderedDict()
    current_section = ""
    section_content = ""

    for page in mathpix_json['pages']:
        for line in page['lines']:
            if '\\section' in line['text']:
                if current_section:
                    sections[current_section] = section_content.strip()
                current_section = line['text'].replace('\\section*{', '').replace('}', '').strip()
                section_content = ""
            else:
                section_content += line['text'] + " "

    if current_section:
        sections[current_section] = section_content.strip()

    return sections

# FAISS related functions
def create_faiss_index(dimension: int):
    index = faiss.IndexFlatL2(dimension)
    return faiss.IndexIDMap(index)

def add_to_faiss_index(index, embeddings: np.ndarray, ids: List[int]):
    index.add_with_ids(embeddings, np.array(ids))

def search_faiss_index(index, query_embedding: np.ndarray, k: int):
    return index.search(query_embedding, k)

# Main functions
def create_paper_embeddings(paper_folder: str, model, index):
    papers_mathpixed = glob.glob(f"{paper_folder}/*/*lines.mmd.json")
    paper_data = []
    embeddings = []
    ids = []

    for i, paper_info_json in enumerate(papers_mathpixed):
        paper_name = os.path.basename(os.path.dirname(paper_info_json))
        paper_infos = read_mathpix_json(paper_info_json)
        title, abstract = get_title_and_abstract(paper_infos)
        
        paper_data.append({
            "id": i,
            "paper_name": paper_name,
            "title": title,
            "abstract": abstract,
            "source": paper_info_json
        })
        
        embedding = model.encode(f"{title} {abstract}")
        embeddings.append(embedding)
        ids.append(i)

    add_to_faiss_index(index, np.array(embeddings), ids)
    return paper_data

def create_section_embeddings(paper_folder: str, model, index):
    papers_mathpixed = glob.glob(f"{paper_folder}/*/*lines.mmd.json")
    section_data = []
    embeddings = []
    ids = []

    for paper_info_json in papers_mathpixed:
        paper_name = os.path.basename(os.path.dirname(paper_info_json))
        paper_infos = read_mathpix_json(paper_info_json)
        sections = get_sections(paper_infos)

        for i, (section_title, section_content) in enumerate(sections.items()):
            section_data.append({
                "id": len(section_data),
                "paper_name": paper_name,
                "section": section_title,
                "content": section_content,
                "source": paper_info_json
            })
            
            embedding = model.encode(f"{section_title} {section_content}")
            embeddings.append(embedding)
            ids.append(len(section_data) - 1)

    add_to_faiss_index(index, np.array(embeddings), ids)
    return section_data

def retrieve_relevant_paper(question: str, model, index, paper_data, k=5):
    query_embedding = model.encode(question)
    distances, indices = search_faiss_index(index, np.array([query_embedding]), k)
    return [paper_data[i] for i in indices[0]]

def retrieve_relevant_section(question: str, model, index, section_data, paper_name, k=1):
    query_embedding = model.encode(question)
    distances, indices = search_faiss_index(index, np.array([query_embedding]), k)
    relevant_sections = [section_data[i] for i in indices[0] if section_data[i]["paper_name"] == paper_name]
    return relevant_sections[0] if relevant_sections else None

def review_papers(question: str, papers: List[Dict], pipeline) -> int:
    papers_text = "\n\n".join([f"Paper {i+1}:\nTitle: {paper['title']}\nAbstract: {paper['abstract']}" for i, paper in enumerate(papers)])
    messages = [
        {"role": "system", "content": "You are an AI assistant tasked with selecting the most relevant academic paper for a given question. Analyze the paper titles and abstracts and choose the most relevant one. Respond with ONLY a single number from 0 to 4, where 0 represents the first paper and 4 represents the fifth paper."},
        {"role": "user", "content": f"Question: {question}\n\nPapers:\n{papers_text}\n\nPlease respond with ONLY the number (0-4) of the most relevant paper."},
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=512,
    )
    response = outputs[0]["generated_text"][-1]['content'].strip()
    
    try:
        index = int(response)
        if 0 <= index <= 4:
            return index
        else:
            raise ValueError
    except ValueError:
        print(f"Warning: Invalid response from model: {response}. Defaulting to first paper.")
        return 0


class ModelType(Enum):
    LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
    BLLAVA = "Bllossom/llama-3.1-Korean-Bllossom-Vision-8B"

@dataclass
class ModelConfig:
    model_id: str
    system_prompt: str
    use_vision: bool
    max_new_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.9

MODEL_CONFIGS = {
    ModelType.LLAMA: ModelConfig(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        system_prompt="You are a helpful AI assistant that answers questions about academic papers. Please provide a concise and relevant answer based on the given context.",
        use_vision=False
    ),
    ModelType.BLLAVA: ModelConfig(
        model_id="Bllossom/llama-3.1-Korean-Bllossom-Vision-8B",
        system_prompt="""You are a versatile AI assistant named Bllava, capable of both understanding and generating text as well as interpreting and analyzing images. Your role is to kindly and effectively answer the user's questions, whether they are about text or images, and provide appropriate and helpful responses to all types of queries.

당신은 텍스트를 이해하고 생성하는 것뿐만 아니라 이미지를 해석하고 분석할 수 있는 다재다능한 AI 어시스턴트 블라썸입니다. 사용자의 질문이 텍스트에 관한 것이든 이미지에 관한 것이든 친절하고 효과적으로 답변하며, 모든 유형의 질의에 대해 적절하고 유용한 응답을 제공하는 것이 당신의 역할입니다.""",
        use_vision=True
    )
}

class RAGSystem:
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.config = MODEL_CONFIGS[model_type]
        self.initialize_model()
        
    def initialize_model(self):
        if self.model_type == ModelType.LLAMA:
            self.model = transformers.pipeline(
                "text-generation",
                model=self.config.model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
            self.processor = None
        else:  # BLLAVA
            from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
            self.processor = LlavaNextProcessor.from_pretrained(self.config.model_id)

    def generate_answer(self, question: str, context: str, image_path: str = None) -> str:
        if self.model_type == ModelType.LLAMA:
            return self._generate_llama_answer(question, context)
        else:
            return self._generate_bllava_answer(question, context, image_path)

    def _generate_llama_answer(self, question: str, context: str) -> str:
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
        ]
        outputs = self.model(
            messages,
            max_new_tokens=self.config.max_new_tokens,
        )
        return outputs[0]["generated_text"][-1]['content'].strip()

    def _generate_bllava_answer(self, question: str, context: str, image_path: str = None) -> str:
        instruction = f"{context}\n\n{question}"
        messages = [
            {'role': 'system', 'content': self.config.system_prompt},
            {'role': 'user', 'content': instruction if image_path is None else f"<image>\n{instruction}"}
        ]

        if image_path is None:
            chat_messages = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt',
            )
            
            bos_token = self.processor.tokenizer.bos_token_id
            chat_messages = torch.cat([torch.tensor([[bos_token]]), chat_messages], dim=-1).to(self.model.device)
            
            inputs = {'input_ids': chat_messages}
        else:
            image = Image.open(image_path).convert('RGB')
            chat_messages = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ).to(self.model.device)
            
            inputs = self.processor(
                chat_messages,
                image,
                return_tensors='pt',
            )

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                do_sample=True,
            )

        return self.processor.tokenizer.decode(output[0])

# [Previous utility functions remain unchanged]

def main(paper_folder: str, question: str, model_type: ModelType = ModelType.LLAMA, image_path: str = None):
    # Initialize model and FAISS indexes
    model = SentenceTransformer("nlpai-lab/KoE5")
    paper_index = create_faiss_index(model.get_sentence_embedding_dimension())
    section_index = create_faiss_index(model.get_sentence_embedding_dimension())

    # Create or load embeddings
    paper_data = create_paper_embeddings(paper_folder, model, paper_index)
    section_data = create_section_embeddings(paper_folder, model, section_index)

    # Initialize RAG system with specified model
    rag_system = RAGSystem(model_type)

    # Generate answer without RAG
    answer_without_rag = rag_system.generate_answer(question, "", image_path)
    print(f"Answer without RAG:\n{answer_without_rag}\n")
    print("-" * 50)

    # Retrieve top 5 relevant papers
    relevant_papers = retrieve_relevant_paper(question, model, paper_index, paper_data, k=10)
    print("#"*100)
    print("relevant_papers")
    for p in relevant_papers:
        print(p["paper_name"])
    print("#"*100)
    if not relevant_papers:
        print("No relevant papers found.")
        return

    # Generate answers with RAG for all top 5 papers (before Reviewer LLM)
    answers_before_review = []
    for paper in relevant_papers:
        relevant_section = retrieve_relevant_section(question, model, section_index, section_data, paper['paper_name'])
        if relevant_section:
            answer = rag_system.generate_answer(question, relevant_section['content'], image_path)
            answers_before_review.append((paper['paper_name'], answer))

    if not answers_before_review:
        print("No relevant sections found before Reviewer LLM.")
        return

    print("Answers with RAG (before Reviewer LLM):")
    for paper_name, answer in answers_before_review:
        print(f"Paper: {paper_name}")
        print(f"Answer: {answer}\n")
    print("-" * 50)
    
    # Use Reviewer LLM to select the most relevant paper
    selected_index = review_papers(question, relevant_papers, rag_system.model)
    selected_paper = relevant_papers[selected_index]

    print(f"Selected paper after review: {selected_paper['paper_name']}")
    print(f"Title: {selected_paper['title']}")
    print(f"Abstract: {selected_paper['abstract'][:200]}...")

    relevant_section = retrieve_relevant_section(question, model, section_index, section_data, selected_paper['paper_name'])
    if not relevant_section:
        print("No relevant section found after Reviewer LLM.")
        return

    print(f"\nMost similar section: {relevant_section['section']}")
    print(f"Section content preview: {relevant_section['content'][:200]}...")

    # Generate final answer with RAG
    answer_with_rag_after_review = rag_system.generate_answer(question, relevant_section['content'], image_path)
    print(f"\nQuestion: {question}")
    print(f"Answer with RAG (after Reviewer LLM):\n{answer_with_rag_after_review}")
    
    print("\n" + "=" * 50)
    print("Comparison:")
    print(f"1. Answer without RAG:\n{answer_without_rag}\n")
    print("2. Answers with RAG (before Reviewer LLM):")
    for paper_name, answer in answers_before_review:
        print(f"Paper: {paper_name}")
        print(f"Answer: {answer}\n")
    print(f"3. Answer with RAG (after Reviewer LLM):\n{answer_with_rag_after_review}")

if __name__ == "__main__":
    '''
    # 고추잎추출물의항산화및암세포증식억제효과
    question = "고추잎추출물 관련 암세포 연구가 있어?"

    # 인간 대장암 세포주 HCT116 배양
    question = "인간 대장암 HCT116 세포 NL, NS, and NSP 과 암세포랑 어떤 상관관계가 있어?"

    # 혈중암세포기반암예후예측진단융합기술개발  
    question = "마이크로바이오칩을 활용한 암세포 분리 기술에 대해 설명해줘"

    # 버섯균사체에의한암세포성장억제효과
    question = "버섯의 항암 및 면역효과에 대해 알려줘"


    #꼬시래기산추출물의primary인체전립선암세포증식억제효과
    question = "극적 관찰요법같은 국소적 치료방법 말고 다른 방법으로 항암전이를 막는 연구 알려줘"
    '''

    paper_folder = "mathpixed_data"
    question = "고추잎추출물 관련 암세포 연구가 있어?"
    
    # Example usage with different models
    # For LLAMA model:
    main(paper_folder, question, ModelType.LLAMA)
    
    # For BLLAVA model:
    # main(paper_folder, question, ModelType.BLLAVA, "image.jpg")  # Optional image path for BLLAVA
