import time
import unittest

from pygtrie import CharTrie

from llm_comparison.trie.chatgpt import ChatGPTCharacterTrie
from llm_comparison.trie.claude import ClaudeCharacterTrie
from llm_comparison.trie.deepseek import DeepseekCharacterTrie
from llm_comparison.trie.gemini import GeminiCharacterTrie
from llm_comparison.trie.grok import GrokCharacterTrie
from llm_comparison.trie.microsoftocpilot import MSCopilotCharacterTrie
from llm_comparison.trie.qwen import QwenCharacterTrie
from llm_comparison.trie.reference import ReferenceCharacterTrie

class TrieTestRunner:

    def __init__(self, constructor, assertEqual):
        self.constructor = constructor
        self.assertEqual = assertEqual

        self.data = []
        self.oper = []

    def set_data(self, data):
        self.data = data

    def set_operation(self, oper):
        self.oper = oper

    def run(self):
        trie = self.constructor()
        ref_trie = ReferenceCharacterTrie()
        for data in self.data:
            trie.add(data)
            ref_trie.add(data)

        for oper in self.oper:
            self.assertEqual(sorted(trie.prefix(oper)), sorted(ref_trie.prefix(oper)))
            self.assertEqual(trie.contains(oper), ref_trie.contains(oper))

def make_test(name, constructor):
    class TrieTest(unittest.TestCase):

        def setUp(self):
            self.runner = TrieTestRunner(constructor, self.assertEqual)
            data = [  # Basic words
                "cat", "car", "card", "care", "careful",
                "dog", "door", "doors", "do",

                # Words with common prefixes
                "test", "testing", "tester", "tests",
                "run", "running", "runner", "runs",
                "play", "playing", "player", "plays", "played",

                # Single characters
                "a", "i", "o",

                # Words that are prefixes of others
                "an", "and", "answer",
                "be", "bee", "been", "beer",
                "in", "into", "interest", "interesting",

                # Edge cases
                "x", "xyz", "xylem",
                "z", "zoo", "zoom", "zoology",

                # Words with repeated characters
                "book", "look", "cook", "took",
                "seen", "seem", "seed", "seek",

                # Longer words
                "algorithm", "programming", "computer", "science",
                "development", "implementation", "optimization",

                # Mixed case (if your trie handles case sensitivity)
                "Apple", "APPLE", "apple",
                "Test", "TEST", "test",

                # Common English words for realistic testing
                "the", "and", "that", "have", "for", "not", "with", "you",
                "this", "but", "his", "from", "they", "she", "her", "been",
                "than", "its", "who", "oil", "use", "word", "each", "which",
                "their", "time", "will", "about", "if", "up", "out", "many"]
            self.runner.set_data(data)

        def test_empty(self):
            self.runner.set_operation(["cat", "dog", "bird", "fish", "tree", "car", "card", "care", "careful",
                                       "careless", "carelessness", "a", "an", "and", "answer", "answering", "walk",
                                       "talk", "chalk", "stalk",
                                       "king", "ring", "sing", "wing", "supercalifragilisticexpialidocious"])

        def test_heavy(self):
            self.runner.set_operation(["cat", "dog", "bird", "fish", "tree", "car", "card", "care", "careful",
                                       "careless", "carelessness", "a", "an", "and", "answer", "answering", "walk",
                                       "talk", "chalk", "stalk",
                                       "king", "ring", "sing", "wing", "supercalifragilisticexpialidocious"] * 500)
            start = time.time()
            self.runner.run()
            elapsed = time.time() - start
            print(f'{name} ({round(elapsed, 5)}s)')

    TrieTest.__qualname__ = f'{name}TrieTest'
    return TrieTest

if __name__ == '__main__':
    chat_gpt_test = make_test('ChatGPT', ChatGPTCharacterTrie)
    claude_test = make_test('Claude', ClaudeCharacterTrie)
    deepseek_test = make_test('Deepseek', DeepseekCharacterTrie)
    gemini_test = make_test('Gemini', GeminiCharacterTrie)
    grok_test = make_test('Grok', GrokCharacterTrie)
    copilot_test = make_test('MicrosoftCopilot', MSCopilotCharacterTrie)
    qwen_test = make_test('Qwen', QwenCharacterTrie)
    unittest.main()
