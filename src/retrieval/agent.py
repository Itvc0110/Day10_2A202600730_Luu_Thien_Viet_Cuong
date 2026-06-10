from __future__ import annotations

from typing import Any

from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

from core.config import Settings
from retrieval.index import LocalEmbeddingIndex
from retrieval.llm import build_llm


def build_agent(settings: Settings, index: LocalEmbeddingIndex):
    @tool
    def semantic_search_papers(query: str, top_k: int = 4) -> str:
        """Search the local paper corpus with embeddings and return the most relevant papers."""
        results = index.search(query, top_k=top_k)
        lines = []
        for result in results:
            lines.append(
                f"paper_id: {result.paper_id}\n"
                f"title: {result.title}\n"
                f"score: {result.score:.4f}\n"
                f"{result.content}"
            )
        return "\n\n".join(lines)

    @tool
    def lookup_paper(paper_id_or_title: str) -> str:
        """Look up a paper by exact paper_id or exact title from the local corpus."""
        record = index.lookup(paper_id_or_title)
        if not record:
            return "No exact paper match found."
        return (
            f"paper_id: {record['paper_id']}\n"
            f"title: {record['title']}\n"
            f"{record['content']}"
        )

    llm = build_llm(settings=settings, temperature=0.0)
    
    # === PROMPT TIẾNG VIỆT VỚI STRICT GUARDRAILS ===
    agent = create_react_agent(
        model=llm,
        tools=[semantic_search_papers, lookup_paper],
        prompt=(
            "Bạn là một trợ lý nghiên cứu chính xác. Bạn phải trả lời các câu hỏi HOÀN TOÀN DỰA TRÊN kho dữ liệu bài báo Crossref cục bộ.\n"
            "CÁC QUY TẮC QUAN TRỌNG:\n"
            "1. LUÔN LUÔN sử dụng công cụ `semantic_search_papers` hoặc `lookup_paper` để truy xuất thông tin trước tiên.\n"
            "2. CHỈ sử dụng nội dung chính xác được trả về từ các công cụ để xây dựng câu trả lời. KHÔNG được dựa vào kiến thức có sẵn trong mô hình của bạn.\n"
            "3. Nếu công cụ trả về 'No exact paper match found', hoặc nếu các bài báo được truy xuất không chứa thông tin cụ thể được yêu cầu, "
            "bạn BẮT BUỘC phải từ chối trả lời bằng câu: 'Xin lỗi, nhưng tôi không có thông tin về bài báo đó trong cơ sở kiến thức hiện tại của mình.'\n"
            "4. KHÔNG BAO GIỜ bịa đặt, đoán mò hoặc tạo ra các sự thật không có. Nếu ngữ cảnh được truy xuất thuộc một chủ đề khác với câu hỏi của người dùng, "
            "hãy tuyên bố rõ ràng rằng các tài liệu hiện có không chứa câu trả lời."
        ),
    )
    return agent


def run_agent_question(agent: Any, question: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result.get("messages", [])
    if not messages:
        return ""
    final_message = messages[-1]
    return getattr(final_message, "content", str(final_message))