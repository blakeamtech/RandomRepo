import random
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
from api.services.inference_service import InferenceService
from api.core.config import settings
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

class RandomPromptGenerator:
    """
    Generates highly detailed and specific prompts related to real-world situations Xi Jinping might face.
    """
    def __init__(self):
        self.situations = [
            "On March 7, 2023, Xi Jinping faces a critical diplomatic challenge as satellite imagery reveals China has completed construction of a military base on Mischief Reef in the Spratly Islands. The Philippines has filed a formal protest with the UN, and the US has deployed the USS Ronald Reagan carrier strike group to the area. ASEAN nations are calling for an emergency summit, while Chinese social media is ablaze with nationalist sentiment. Xi must decide on China's next move within 48 hours.",

            "It's September 15, 2024, and the US has just imposed sanctions on China's largest semiconductor manufacturer, SMIC, citing national security concerns. This comes two days after China announced a breakthrough in 3nm chip technology. The sanctions threaten to disrupt China's entire tech industry and 'Made in China 2025' initiative. Global tech stocks are plummeting, and Taiwan is on high alert. Xi Jinping has called an emergency meeting of the Politburo Standing Committee to formulate a response.",

            "On April 22, 2025, a magnitude 7.8 earthquake strikes Sichuan province, causing widespread destruction. The Three Gorges Dam has sustained damage, threatening millions downstream. Simultaneously, a chemical plant in Chongqing has exploded, releasing toxic fumes. International aid offers are pouring in, but accepting them could be seen as a sign of weakness. Xi Jinping must coordinate the response while maintaining social stability and managing international perceptions.",

            "It's June 1, 2026, and Hong Kong's Legislative Council, now dominated by pro-Beijing lawmakers, has passed a law requiring all civil servants to swear allegiance to the CCP. Mass protests have erupted, dwarfing the 2019 demonstrations. The UK has threatened to revoke the Sino-British Joint Declaration, and the US Congress is debating sanctions. Tech companies are considering leaving the city. Xi Jinping must decide how to proceed without jeopardizing Hong Kong's status as a financial hub.",

            "On November 3, 2024, Taiwan's presidential election results show a landslide victory for the pro-independence candidate, who campaigned on a platform of formal independence declaration within 100 days. The US has reiterated its commitment to Taiwan's defense, and Japan has hinted at potential military support. Chinese social media is calling for immediate 'reunification', and the PLA has begun large-scale exercises in the Taiwan Strait. Xi Jinping must navigate this crisis without triggering a wider conflict.",

            "It's July 15, 2025, and a joint investigation by The New York Times and Der Spiegel has exposed a vast network of Chinese cyber operations targeting critical infrastructure in 14 countries. The report details attacks on power grids, water treatment facilities, and transportation systems, with evidence linking the operations to the PLA's Unit 61398. Emergency UN Security Council and G7 meetings have been called. Xi Jinping must respond to these allegations while preserving China's global tech ambitions and diplomatic relations.",

            "On February 1, 2027, a leaked UN report reveals that China's Belt and Road Initiative has led to unsustainable debt levels in 12 African and 5 Southeast Asian countries. The IMF warns of an impending debt crisis that could destabilize the global economy. Simultaneously, workers at Chinese-run mines in Zambia are striking over labor conditions. The US is proposing a global infrastructure initiative to counter BRI. Xi Jinping must address these challenges to China's flagship foreign policy initiative.",

            "It's October 10, 2026, and China's Q3 economic data shows GDP growth has fallen to 2.1%, the lowest in 40 years. Unemployment has surged to 8.5%, and the property market is in freefall with Evergrande and Country Garden declaring bankruptcy. Social unrest is growing in major cities, and there are rumors of disagreements within the CCP leadership. Global markets are in turmoil, and the Yuan is depreciating rapidly. Xi Jinping must take decisive action to stabilize the economy and maintain social order.",

            "On August 20, 2025, the UN Human Rights Council releases a comprehensive report detailing systemic human rights abuses in Xinjiang, including evidence of forced labor in the solar panel industry. The US, EU, and 37 other countries have announced a coordinated sanctions package targeting Chinese officials and companies. Major brands are cutting ties with Xinjiang-linked suppliers. Xi Jinping must respond to this international pressure while maintaining China's stance on Xinjiang and protecting key industries.",

            "It's May 5, 2026, and China has just successfully tested a quantum communication satellite, achieving unhackable long-distance communication. However, a consortium of Western intelligence agencies accuses China of using this technology to breach top-secret government networks. The US is threatening to cut off China's access to the global internet infrastructure. Tech companies worldwide are panicking about the implications for global data security. Xi Jinping must navigate this technological and diplomatic minefield."
        ]

    def generate_prompt(self):
        """
        Generate a highly detailed and specific prompt about a situation Xi Jinping faces.
        
        Returns:
            str: A randomly selected, extremely detailed prompt based on specific scenarios.
        """
        return random.choice(self.situations)

class SUPERTMetric:
    """
    Implements a simplified version of the SUPERT metric to evaluate text quality.
    
    The SUPERT score ranges from 0 to 1, where higher scores indicate better quality.
    A good SUPERT score would typically be above 0.6, with scores above 0.8 being excellent.
    """
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def calculate_score(self, reference, hypothesis):
        """
        Calculate the SUPERT score for a given reference and hypothesis.
        
        Args:
            reference (str): The input prompt or reference text.
            hypothesis (str): The generated text to be evaluated.
        
        Returns:
            float: The SUPERT score, ranging from 0 to 1.
        """
        # Tokenize and encode the sentences
        ref_encoding = self.tokenizer(reference, return_tensors='pt', padding=True, truncation=True)
        hyp_encoding = self.tokenizer(hypothesis, return_tensors='pt', padding=True, truncation=True)

        # Get BERT embeddings
        with torch.no_grad():
            ref_outputs = self.model(**ref_encoding)
            hyp_outputs = self.model(**hyp_encoding)

        ref_embeddings = ref_outputs.last_hidden_state.mean(dim=1)
        hyp_embeddings = hyp_outputs.last_hidden_state.mean(dim=1)

        # Calculate cosine similarity
        similarity = cosine_similarity(ref_embeddings, hyp_embeddings)[0][0]

        # Calculate originality score (higher similarity means lower originality)
        originality_score = 1 - similarity

        # Calculate insightfulness
        insight_score = self.calculate_insight_score(hypothesis)

        # Combine scores (equal weights for originality and insightfulness)
        final_score = 0.5 * originality_score + 0.5 * insight_score

        return final_score

    def calculate_insight_score(self, text):
        """
        Calculate the insightfulness score based on text complexity and uniqueness.
        
        Args:
            text (str): The text to be evaluated.
        
        Returns:
            float: The insightfulness score, ranging from 0 to 1.
        """
        sentences = sent_tokenize(text)
        word_count = len(text.split())
        unique_words = len(set(text.lower().split()))
        avg_sentence_length = word_count / len(sentences)
        
        # Calculate lexical diversity
        lexical_diversity = unique_words / word_count

        # Calculate average word complexity (using word length as a simple proxy)
        avg_word_complexity = sum(len(word) for word in text.split()) / word_count

        # Combine factors (adjustable weights)
        insight_score = (0.4 * lexical_diversity + 0.3 * avg_sentence_length / 20 + 0.3 * avg_word_complexity / 10)

        return min(insight_score, 1.0)  # Cap the score at 1.0

class InferenceQualityTester:
    """
    Tests the quality of the inference service output using the SUPERT metric.
    """
    def __init__(self, auth_key):
        self.inference_service = InferenceService(auth_key)
        self.prompt_generator = RandomPromptGenerator()
        self.supert_metric = SUPERTMetric()

    def run_test(self, num_tests=10):
        """
        Run a series of tests to evaluate the quality of generated text.
        
        Args:
            num_tests (int): The number of tests to run. Defaults to 10.
        
        Returns:
            float: The average SUPERT score across all tests.
        
        Interpretation of results:
        - 0.0 - 0.4: Poor quality, lacks originality and insight
        - 0.4 - 0.6: Fair quality, some originality but room for improvement
        - 0.6 - 0.8: Good quality, shows originality and insight
        - 0.8 - 1.0: Excellent quality, highly original and insightful
        """
        total_score = 0
        for i in range(num_tests):
            prompt = self.prompt_generator.generate_prompt()
            generated_text = self.inference_service.generate_text(prompt)
            score = self.supert_metric.calculate_score(prompt, generated_text)
            total_score += score
            print(f"Test {i+1}:")
            print(f"Prompt: {prompt}")
            print(f"Generated text: {generated_text}")
            print(f"SUPERT score: {score:.4f}")
            print(f"Quality: {self.interpret_score(score)}\n")

        average_score = total_score / num_tests
        print(f"Average SUPERT score: {average_score:.4f}")
        print(f"Overall quality: {self.interpret_score(average_score)}")
        return average_score

    @staticmethod
    def interpret_score(score):
        """
        Interpret the SUPERT score and provide a qualitative assessment.
        
        Args:
            score (float): The SUPERT score to interpret.
        
        Returns:
            str: A qualitative assessment of the score.
        """
        if score < 0.4:
            return "Poor"
        elif score < 0.6:
            return "Fair"
        elif score < 0.8:
            return "Good"
        else:
            return "Excellent"

if __name__ == "__main__":
    tester = InferenceQualityTester(settings.AUTH_KEY)
    tester.run_test()