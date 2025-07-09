import random
import time
import unittest
from functools import cmp_to_key

from llm_comparison.red_black_tree.chatgpt import ChatGPTRBT
from llm_comparison.red_black_tree.claude import ClaudeRBT
from llm_comparison.red_black_tree.deepseek import DeepseekRBT
from llm_comparison.red_black_tree.gemini import GeminiRBT
from llm_comparison.red_black_tree.grok import GrokRBT
from llm_comparison.red_black_tree.microsoftcopilot import MSCopilotRBT
from llm_comparison.red_black_tree.qwen import QwenRBT

ADD = 1
REMOVE = 2
REMOVE_FIRST = 3
CLEAR = 4

class TestPeople:

    def __init__(self, name=None, age=0):
        self.name = name
        self.age = age

    def __eq__(self, other):
        return self.name == other.name and self.age == other.age

class RBTTestRunner:

    def __init__(self, constructor, comparator, assertEqual, assertRaises):
        self.constructor = constructor
        self.assertEqual = assertEqual
        self.assertRaises = assertRaises

        self.comparator = comparator
        self.data = []
        self.oper = []

    def _remove(self, lst, item):
        for i, element in enumerate(lst):
            if self.comparator(element, item) == 0:
                return lst.pop(i)

    def set_data(self, *data):
        self.data = data
        return self

    def set_operation(self, *oper):
        self.oper = [(oper[i], oper[i + 1]) for i in range(0, len(oper), 2)]
        return self

    def run(self):
        tree = self.constructor(self.comparator)
        assertEqual = self.assertEqual
        assertRaises = self.assertRaises

        tree_data = []
        tree_size = 0

        assertEqual(tree_size, 0)

        for oper, i in self.oper:
            item = self.data[i]
            if oper == ADD:
                if any(self.comparator(i, item) == 0 for i in tree_data):
                    with assertRaises(ValueError):
                        tree.add(item)
                    assertEqual(tree.contains(item), True)
                    assertEqual(tree.size(), tree_size)
                else:
                    tree.add(item)
                    tree_size += 1
                    tree_data.append(item)
                    assertEqual(tree.contains(item), True)
                    assertEqual(tree.size(), tree_size)
            elif oper == REMOVE:
                if not any(self.comparator(i, item) == 0 for i in tree_data):
                    with assertRaises(ValueError):
                        tree.remove(item)
                else:
                    removed_item = tree.remove(item)
                    expected_item = self._remove(tree_data, item)
                    tree_size -= 1
                    assertEqual(removed_item, expected_item)
                    assertEqual(tree.contains(removed_item), False)
                    assertEqual(tree.size(), tree_size)
            elif oper == REMOVE_FIRST:
                if tree_size == 0:
                    with assertRaises(ValueError):
                        tree.remove_first()
                else:
                    tree_data.sort(key=cmp_to_key(self.comparator))
                    removed_item = tree.remove_first()
                    expected_item = tree_data.pop(0)
                    tree_size -= 1
                    assertEqual(removed_item, expected_item)
                    assertEqual(tree.contains(removed_item), False)
                    assertEqual(tree.size(), tree_size)
            elif oper == CLEAR:
                tree.clear()
                tree_size = 0
                tree_data = []
                assertEqual(tree.size(), 0)

def make_test(name, constructor):
    class RBTTest(unittest.TestCase):

        def setUp(self):
            def comparator(x, y):
                return x.age - y.age

            names = ['Ella', 'Narin', 'Wonhee', 'Iroha', 'Leeseo', 'Eunchae', 'Anna', 'Wonyoung', 'Liz', 'Sooin',
                     'Gawon', 'Moka', 'Yujin', 'Rei', 'Kazuha', 'Yunah', 'Minju', 'Gaeul', 'Ningning', 'Yunjin',
                     'Chaewon', 'Giselle', 'Winter', 'Karina', 'Sakura']
            ages = [16, 17, 17, 17, 18, 18, 19, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 22, 22, 23, 24, 24, 24, 25, 27]

            data = [TestPeople(name=name, age=age) for name, age in zip(names, ages)]

            self.runner = RBTTestRunner(constructor, comparator, self.assertEqual, self.assertRaises)
            self.runner.set_data(*data)

        def test_add_all(self):
            self.runner.set_operation(ADD, 0, ADD, 1, ADD, 2, ADD, 3, ADD, 4, ADD, 5, ADD, 6, ADD, 7, ADD, 8, ADD, 9,
                                      ADD, 10, ADD, 11, ADD, 12, ADD, 13, ADD, 14, ADD, 15, ADD, 16, ADD, 17, ADD, 18,
                                      ADD, 19, ADD, 20, ADD, 21, ADD, 22, ADD, 23, ADD, 24, CLEAR, 0)
            self.runner.run()

        def test_remove_all(self):
            self.runner.set_operation(REMOVE, 0, REMOVE, 1, REMOVE, 2, REMOVE, 3, REMOVE, 4, REMOVE, 5, REMOVE, 6,
                                      REMOVE, 7, REMOVE, 8, REMOVE, 9, REMOVE, 10, REMOVE, 11, REMOVE, 12, REMOVE, 13,
                                      REMOVE, 14, REMOVE, 15, REMOVE, 16, REMOVE, 17, REMOVE, 18, REMOVE, 19, REMOVE,
                                      20, REMOVE, 21, REMOVE, 22, REMOVE, 23, REMOVE, 24, CLEAR, 0)
            self.runner.run()

        def test_add_remove_all(self):
            self.runner.set_operation(ADD, 0, ADD, 1, ADD, 2, ADD, 3, ADD, 4, ADD, 5, ADD, 6, ADD, 7, ADD, 8, ADD, 9,
                                      ADD, 10, ADD, 11, ADD, 12, ADD, 13, ADD, 14, ADD, 15, ADD, 16, ADD, 17, ADD, 18,
                                      ADD, 19, ADD, 20, ADD, 21, ADD, 22, ADD, 23, ADD, 24, REMOVE, 0, REMOVE, 1,
                                      REMOVE, 2, REMOVE, 3, REMOVE, 4, REMOVE, 5, REMOVE, 6, REMOVE, 7, REMOVE, 8,
                                      REMOVE, 9, REMOVE, 10, REMOVE, 11, REMOVE, 12, REMOVE, 13, REMOVE, 14, REMOVE, 15,
                                      REMOVE, 16, REMOVE, 17, REMOVE, 18, REMOVE, 19, REMOVE, 20, REMOVE, 21, REMOVE,
                                      22, REMOVE, 23, REMOVE, 24, CLEAR, 0)
            self.runner.run()

        def test_remove_first(self):
            self.runner.set_operation(REMOVE_FIRST, 0, ADD, 0, REMOVE_FIRST, 0, ADD, 1, ADD, 2, REMOVE_FIRST, 0, ADD,
                                      3, ADD, 4, ADD, 5, REMOVE_FIRST, 0, ADD, 6, ADD, 7, ADD, 8, ADD, 9, REMOVE_FIRST,
                                      0, ADD, 10, ADD, 11, ADD, 12, ADD, 13, ADD, 14, REMOVE_FIRST, 0, ADD, 15, ADD,
                                      16, ADD, 17, ADD, 18, ADD, 19, ADD, 20, ADD, 21, ADD, 22, ADD, 23, ADD, 24,
                                      REMOVE_FIRST, 0, REMOVE_FIRST, 0, CLEAR, 0)
            self.runner.run()

        def test_stress(self):
            self.runner = RBTTestRunner(constructor, lambda x, y: x - y, self.assertEqual, self.assertRaises)
            self.runner.set_data(*range(1000))
            start = time.time()
            self.runner.set_operation(*operations).run()
            elapsed = time.time() - start
            print(f'{name} ({round(elapsed, 5)}s)')

    RBTTest.__qualname__ = f'{name}RBTTest'

    return RBTTest

if __name__ == '__main__':
    operations = []
    oper = ['ADD', 'REMOVE', 'REMOVE_FIRST', 'CLEAR']
    for i in range(20000):
        operations.append(oper[random.randint(0, 3)])
        operations.append(random.randint(0, 999))

    chat_gpt_test = make_test('ChatGPT', ChatGPTRBT)
    claude_test = make_test('Claude', ClaudeRBT)
    deepseek_test = make_test('Deepseek', DeepseekRBT)
    gemini_test = make_test('Gemini', GeminiRBT)
    grok_test = make_test('Grok', GrokRBT)
    copilot_test = make_test('MicrosoftCopilot', MSCopilotRBT)
    qwen_test = make_test('Qwen', QwenRBT)

    unittest.main()
