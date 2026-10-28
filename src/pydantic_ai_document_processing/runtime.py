from pydantic_ai import BinaryContent
from typing import Any
from pydantic_ai_document_processing.schemas import (
    ProcessResult,
    DocumentClassification,
)
from pydantic_ai_document_processing.agent import (
    DocumentSchema,
    chromadb_collection,
    embedder,
    FEW_SHOT_EXAMPLE_QTY,
    classification_agent,
    ClassificationAgentDeps,
    extraction_agent_schema,
    ExtractionAgentDeps,
    extraction_agent,
)
import pymupdf
from chromadb import Knn
import io
from pathlib import Path
from markitdown import Markitdown

md = Markitdown()


# function to process a pdf document
async def process_document(pdf_path: Path) -> ProcessResult:
    try:
        # open pdf file
        doc = pymupdf.open(pdf_path.as_posix())

        if len(doc) == 0:
            raise ValueError("No pages found in PDF")

        single_page_doc = pymupdf.open()
        single_page_doc.insert_pdf(doc, from_page=0, to_page=1)

        pdf_bytes = single_page_doc.tobytes()
        single_page_doc.close()
        doc.close()

        # document to markdown
        page_md = md.convert(io.BytesIO(pdf_bytes), file_extension=".pdf")

        retrieval_query = page_md.text_content

        retrieval_results = await _retrieval_helper(
            retrieval_query, FEW_SHOT_EXAMPLE_QTY
        )

        few_shot_examples = [
            {
                "document": retrieval_results["documents"][0][m],
                "category": retrieval_results["metadatas"][0]["category"],
            }
            for m in range(FEW_SHOT_EXAMPLE_QTY)
        ]

        classification_result = await classification_agent.run(
            [BinaryContent(data=pdf_path.read_bytes(), media_type="application/pdf")],
            deps=ClassificationAgentDeps(few_shot_examples=few_shot_examples),
        )

        # validate classification result exists
        if classification_result.output:
            doc_type = classification_result.output.document_type
            doc_schema = extraction_agent_schema(doc_type)
        else:
            raise ValueError("No classification result found")

        extraction_result = await extraction_agent.run(
            [BinaryContent(data=pdf_path.read_bytes(), media_type="application/pdf")],
            deps=ExtractionAgentDeps(doc_type=doc_type, doc_schema=doc_schema),
            output_type=doc_schema,
        )

        if extraction_result.output:
            return ProcessResult(
                document_type=doc_type,
                document_data=extraction_result.output,
                success=True,
                error_message=None,
            )
        else:
            return ProcessResult(
                document_type=doc_type,
                document_data=None,
                success=False,
                error_message="No extraction result found",
            )

    except Exception as e:
        return ProcessResult(
            success=False, error_message=f"Error processing document: {e}"
        )


# retrieval helper
async def _retrieval_helper(query: str, k: int = 3) -> dict[str, Any]:
    try:
        # embed document
        emb_result = await embedder.embed_documents(query)

        emb = emb_result.embeddings[0]

        # retrieve documents
        results = await chromadb_collection.query(
            query_embeddings=[emb],
            n_results=k,
        )
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return {}

    return results
