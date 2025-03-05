import json
import logging
import re

# Configure logging
logging.basicConfig(filename="rag_system.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Synthesizer:
    @staticmethod
    def generate_response(question, context):
        """Generate a clean JSON response from multiple database results."""
        try:
            if not context or not isinstance(context, list) or len(context) == 0:
                logging.warning(f"⚠️ No answer found for: {question}")
                return {"status": "error", "message": "No relevant answers found"}

            # ✅ Extract multiple answers
            answers = []
            categories = set()

            for answer_data in context:
                raw_answer = answer_data.get("content", "").strip()
                cleaned_answer = re.sub(r"^Question:.*?\nAnswer:\s*", "", raw_answer).strip()

                if cleaned_answer:
                    answers.append(cleaned_answer)
                    categories.add(answer_data.get("category", "General"))

            response = {
                "status": "success",
                "question": question,
                "answer": " ".join(answers),  # ✅ Combine multiple answers
                "category": ", ".join(categories)  # ✅ Show multiple categories if needed
            }

            logging.info(f"✅ Generated response: {response}")
            return response
        except Exception as e:
            logging.error(f"❌ Error generating response: {str(e)}")
            return {"status": "error", "message": "Internal error"}
