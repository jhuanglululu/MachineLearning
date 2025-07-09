from typing import List

class CharacterTrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class QwenCharacterTrie:
    def __init__(self):
        self.root = CharacterTrieNode()

    def add(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = CharacterTrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def contains(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def prefix(self, word: str) -> List[str]:
        node = self.root
        for char in word:
            if char not in node.children:
                return []
            node = node.children[char]

        results = []

        def dfs(current_node, current_prefix):
            if current_node.is_end_of_word:
                results.append(current_prefix)
            for char, child_node in current_node.children.items():
                dfs(child_node, current_prefix + char)

        dfs(node, word)
        return results