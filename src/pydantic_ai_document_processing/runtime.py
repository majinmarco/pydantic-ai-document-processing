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
from markitdown import MarkItDown
import asyncio
import logfire

md = MarkItDown()

logfire.configure(scrubbing = False)

# Indexed pdf_path values were stored relative to the vectorization notebook.
_INDEX_PATH_BASE = Path(__file__).resolve().parent / "notebooks"


def _resolve_indexed_pdf_path(stored_path: str) -> Path:
    path = Path(stored_path)
    if path.is_absolute():
        return path
    return (_INDEX_PATH_BASE / path).resolve()


# function to process a pdf document
async def process_document(pdf_path: Path) -> ProcessResult:
    try:
        logfire.log(
            "info", "Processing document: {pdf_path}", {"pdf_path": str(pdf_path)}
        )

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
                "document_path": str(
                    _resolve_indexed_pdf_path(
                        retrieval_results["metadatas"][0][m]["pdf_path"]
                    )
                ),
                "category": retrieval_results["metadatas"][0][m]["category"],
            }
            for m in range(FEW_SHOT_EXAMPLE_QTY)
        ]

        classification_user_messages = await classification_prompt_helper(
            few_shot_examples, pdf_path
        )

        classification_result = await classification_agent.run(
            classification_user_messages,
            deps=ClassificationAgentDeps(),
        )

        # validate classification result exists
        if classification_result.output:
            logfire.log(
                "info",
                "Classification result: {classification_result}",
                {"classification_result": classification_result.output},
            )
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
            logfire.log(
                "info",
                "Extraction result: {extraction_result}",
                {"extraction_result": extraction_result.output},
            )
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
        logfire.log("error", "Error processing document: {e}", {"e": str(e)})
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
        results = chromadb_collection.query(
            query_embeddings=[emb],
            n_results=k,
        )

        logfire.log(
            "info",
            "Retrieval results: {results}. Keys: {keys}",
            {"results": results, "keys": results.keys()},
        )
        logfire.log(
            "info",
            "Metadatas: {metadatas}, Documents: {data}",
            {"metadatas": results["metadatas"], "data": results["data"]},
        )
        logfire.log(
            "info", "Metadata keys: {keys}", {"keys": results["metadatas"][0][0].keys()}
        )
        logfire.log(
            "info",
            "Metadata length: {length}",
            {"length": len(results["metadatas"][0])},
        )
    except Exception as e:
        logfire.log("error", "Error retrieving documents: {e}", {"e": str(e)})
        return {}

    return results


async def classification_prompt_helper(
    few_shot_examples: list[dict[str, str]], input_document_path: Path
) -> list[BinaryContent | str]:
    # each few-shot document is a binary content, followed by the category,
    # ended by the actual document to classify
    classification_user_messages = [
        [
            BinaryContent(
                data=Path(example["document_path"]).read_bytes(),
                media_type="application/pdf",
            ),
            f"document_type:{example['category']}.",
        ]
        for example in few_shot_examples
    ]

    # convert to list of messages instead of list of lists
    classification_user_messages = [
        message for sublist in classification_user_messages for message in sublist
    ]

    # add the actual document to classify
    classification_user_messages.append(
        BinaryContent(
            data=input_document_path.read_bytes(), media_type="application/pdf"
        )
    )

    return classification_user_messages


if __name__ == "__main__":
    pdf_path = (
        Path(__file__).resolve().parent
        / "data/company-document-dataset-v2/train/invoice/0a0b4bd2e80b16e2.pdf"
    )
    asyncio.run(process_document(pdf_path))
