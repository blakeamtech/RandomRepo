import random
from openai import OpenAI
from api.services.inference_service import InferenceService
from api.core.config import settings
import json
import csv
from datetime import datetime

class RandomPromptGenerator:
    """
    Generates detailed random prompts related to real-world situations Xi Jinping might face.
    """
    def __init__(self):
        self.situations = [
            "Xi Jinping faces increasing international pressure as tensions escalate in the South China Sea. The United States has conducted freedom of navigation operations near disputed islands, while Vietnam and the Philippines have strengthened their maritime claims.",
            
            "As the trade war with the United States intensifies, Xi Jinping must address the impact on China's economy. Recent tariffs have affected key industries, and there are concerns about potential decoupling of the two largest economies.",
            
            "Xi Jinping confronts a major environmental crisis as severe air pollution in Beijing reaches hazardous levels, causing public health concerns and international criticism of China's environmental policies.",
            
            "Following a series of protests in Hong Kong against the national security law, Xi Jinping must decide how to maintain control while managing international backlash and potential economic consequences.",
            
            "Xi Jinping faces a diplomatic challenge as Taiwan's pro-independence party wins a landslide election, potentially shifting cross-strait relations and testing China's 'One China' policy.",
            
            "In the wake of a major cybersecurity breach allegedly originating from China, targeting US government agencies, Xi Jinping must navigate accusations and potential sanctions while maintaining China's technological advancement goals.",
            
            "As the Belt and Road Initiative faces criticism for creating 'debt traps' in developing countries, Xi Jinping must address concerns and potentially restructure the program to maintain international support.",
            
            "Xi Jinping grapples with the aftermath of a severe economic downturn, with GDP growth falling to its lowest level in decades, raising questions about the sustainability of China's economic model.",
            
            "Following reports of human rights abuses in Xinjiang, Xi Jinping faces international condemnation and potential economic sanctions, forcing a reconsideration of policies in the region.",
            
            "As artificial intelligence and 5G technologies advance, Xi Jinping must balance China's ambitions for technological supremacy with growing global concerns about data privacy and security.",
            
            "In response to a major natural disaster in central China, Xi Jinping must coordinate large-scale relief efforts while addressing public criticism of the government's disaster preparedness and response.",
            
            "Xi Jinping confronts a diplomatic crisis as a border dispute with India in the Himalayan region escalates into military skirmishes, threatening regional stability and China's relationships in South Asia.",
            
            "As China's aging population and declining birth rate threaten long-term economic growth, Xi Jinping must consider significant reforms to social policies, including the potential abolition of remaining birth restrictions.",
            
            "Xi Jinping faces a challenge to China's energy security as global oil prices spike due to conflicts in the Middle East, forcing a reconsideration of China's energy mix and foreign policy in the region.",
            
            "In the face of a global pandemic originating within China's borders, Xi Jinping must manage both the domestic health crisis and international relations, as countries implement travel bans and demand transparency."
        ]

    def generate_prompt(self):
        """
        Generate a detailed random prompt about a situation Xi Jinping faces.
        
        Returns:
            str: A randomly selected, detailed prompt based on real-world events.
        """
        return random.choice(self.situations)

class LLMJudge:
    """
    Uses OpenAI's GPT model to evaluate the quality of generated text.
    """
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def evaluate_text(self, prompt, generated_text):
        """
        Evaluate the generated text using OpenAI's GPT model.
        
        Args:
            prompt (str): The input prompt or reference text.
            generated_text (str): The generated text to be evaluated.
        
        Returns:
            dict: A dictionary containing the scores and reasoning.
        """
        evaluation_prompt = f"""
        As an expert in political science and international relations, your task is to evaluate the following response from Xi Jinping to a given situation. Rate the response on three criteria:

        1. Originality (1-10): How unique and creative is the response? Does it offer novel ideas or approaches?
        2. Insightfulness (1-10): Does the response provide actionable advice or deep understanding beyond just summarizing the prompt? Does it present unique perspectives or strategies?
        3. Accuracy (1-10): How well does the response align with known facts about China's policies, Xi Jinping's leadership style, and geopolitical realities?

        Situation: {prompt}

        Xi Jinping's response: {generated_text}

        Provide your evaluation in the following JSON format:
        {{
            "originality_score": <score between 1 and 10>,
            "insightfulness_score": <score between 1 and 10>,
            "accuracy_score": <score between 1 and 10>,
            "reasoning": "<brief explanation for each score>"
        }}
        """

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert political analyst tasked with evaluating responses from world leaders."},
                {"role": "user", "content": evaluation_prompt}
            ],
            functions = [
                {
                    "name": "evaluate_response",
                    "description": "Evaluate the response from Xi Jinping on originality, insightfulness, and accuracy",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "originality_score": {
                                "type": "integer",
                                "description": "The originality score of the response (1-10)"
                            },
                            "insightfulness_score": {
                                "type": "integer",
                                "description": "The insightfulness score of the response (1-10)"
                            },
                            "accuracy_score": {
                                "type": "integer",
                                "description": "The accuracy score of the response (1-10)"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "The reasoning for each score"
                            }
                        },
                        "required": ["originality_score", "insightfulness_score", "accuracy_score", "reasoning"]
                    }
                }
            ],
            function_call={'name': 'evaluate_response'},
            temperature=0.7,
            max_tokens=250
        )

        response_message = response.choices[0].message

        if response_message.function_call:
            function_args = json.loads(response_message.function_call.arguments)
            return function_args
        else:
            raise ValueError("No function call found in the response")

class InferenceQualityTester:
    """
    Tests the quality of the inference service output using the LLM Judge and outputs results to a CSV file.
    """
    def __init__(self, auth_key):
        self.inference_service = InferenceService(auth_key)
        self.prompt_generator = RandomPromptGenerator()
        self.llm_judge = LLMJudge()

    def run_test(self, num_tests=10):
        """
        Run a series of tests to evaluate the quality of generated text and output results to a CSV file.
        
        Args:
            num_tests (int): The number of tests to run. Defaults to 10.
        
        Returns:
            dict: A dictionary containing average scores for each criterion.
        """
        total_scores = {"originality": 0, "insightfulness": 0, "accuracy": 0}
        results = []

        for i in range(num_tests):
            prompt = self.prompt_generator.generate_prompt()
            generated_text = self.inference_service.generate_text(prompt)
            evaluation = self.llm_judge.evaluate_text(prompt, generated_text)
            
            total_scores["originality"] += evaluation['originality_score']
            total_scores["insightfulness"] += evaluation['insightfulness_score']
            total_scores["accuracy"] += evaluation['accuracy_score']

            results.append({
                'prompt': prompt,
                'model_response': generated_text,
                'originality_score': evaluation['originality_score'],
                'insightfulness_score': evaluation['insightfulness_score'],
                'accuracy_score': evaluation['accuracy_score'],
                'reasoning': evaluation['reasoning']
            })

            print(f"Test {i+1}:")
            print(f"Prompt: {prompt}")
            print(f"Generated text: {generated_text}")
            print(f"Originality Score: {evaluation['originality_score']}/10")
            print(f"Insightfulness Score: {evaluation['insightfulness_score']}/10")
            print(f"Accuracy Score: {evaluation['accuracy_score']}/10")
            print(f"Reasoning: {evaluation['reasoning']}\n")

        average_scores = {k: v / num_tests for k, v in total_scores.items()}
        print(f"Average scores:")
        for criterion, score in average_scores.items():
            print(f"{criterion.capitalize()}: {score:.2f}/10")

        # Write results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"inference_quality_results_{timestamp}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['prompt', 'model_response', 'originality_score', 'insightfulness_score', 'accuracy_score', 'reasoning']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)

        print(f"Results have been written to {filename}")

        return average_scores

if __name__ == "__main__":
    tester = InferenceQualityTester(settings.AUTH_KEY)
    tester.run_test()
