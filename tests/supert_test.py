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
            dict: A dictionary containing the score and reasoning.
        """
        evaluation_prompt = f"""
        As an expert in political science and international relations, your task is to evaluate the following response from Xi Jinping to a given situation. Rate the response on a scale of 1 to 10, where 1 is poor and 10 is excellent. Consider factors such as strategic thinking, diplomatic nuance, and alignment with China's interests.

        Situation: {prompt}

        Xi Jinping's response: {generated_text}

        Provide your evaluation in the following JSON format:
        {{
            "score": <score between 1 and 10>,
            "reasoning": "<brief explanation for the score>"
        }}
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert political analyst tasked with evaluating responses from world leaders."},
                {"role": "user", "content": evaluation_prompt}
            ],
            functions = [
                {
                    "name": "evaluate_response",
                    "description": "Evaluate the response from Xi Jinping on a scale of 1 to 10",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "score": {
                                "type": "integer",
                                "description": "The score of the response"
                            },  
                            "reasoning": {
                                "type": "string",
                                "description": "The reasoning for the score"
                            }
                        },
                        "required": ["score", "reasoning"]
                    }
                }
            ],
            function_call={'name': 'evaluate_response'},
            temperature=0.7,
            max_tokens=150
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
            float: The average score across all tests.
        """
        total_score = 0
        results = []

        for i in range(num_tests):
            prompt = self.prompt_generator.generate_prompt()
            generated_text = self.inference_service.generate_text(prompt)
            evaluation = self.llm_judge.evaluate_text(prompt, generated_text)
            score = evaluation['score']
            total_score += score

            results.append({
                'prompt': prompt,
                'model_response': generated_text,
                'model_rating': score,
                'model_reasoning': evaluation['reasoning']
            })

            print(f"Test {i+1}:")
            print(f"Prompt: {prompt}")
            print(f"Generated text: {generated_text}")
            print(f"Score: {score}/10")
            print(f"Reasoning: {evaluation['reasoning']}\n")

        average_score = total_score / num_tests
        print(f"Average score: {average_score:.2f}/10")

        # Write results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"inference_quality_results_{timestamp}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['prompt', 'model_response', 'model_rating', 'model_reasoning']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)

        print(f"Results have been written to {filename}")

        return average_score

if __name__ == "__main__":
    tester = InferenceQualityTester(settings.AUTH_KEY)
    tester.run_test()
