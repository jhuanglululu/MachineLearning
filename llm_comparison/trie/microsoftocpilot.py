class MSCopilotCharacterTrie:
    class TrieNode:
        def __init__(self):
            self.children = {}  # maps characters to TrieNode
            self.is_end_of_word = False

    def __init__(self):
        self.root = self.TrieNode()

    def add(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = self.TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def contains(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def prefix(self, word: str) -> list[str]:
        def collect_words(node: MSCopilotCharacterTrie.TrieNode, prefix: str) -> list[str]:
            words = []
            if node.is_end_of_word:
                words.append(prefix)
            for char, child in node.children.items():
                words.extend(collect_words(child, prefix + char))
            return words

        node = self.root
        for char in word:
            if char not in node.children:
                return []  # no words with given prefix
            node = node.children[char]
        return collect_words(node, word)