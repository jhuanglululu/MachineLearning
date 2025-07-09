class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class ChatGPTCharacterTrie:
    def __init__(self):
        self.root = TrieNode()

    def add(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def contains(self, word: str) -> bool:
        node = self._find_node(word)
        return node is not None and node.is_end_of_word

    def prefix(self, word: str) -> list[str]:
        node = self._find_node(word)
        result = []

        def dfs(current_node: TrieNode, path: str):
            if current_node.is_end_of_word:
                result.append(path)
            for char, child in current_node.children.items():
                dfs(child, path + char)

        if node:
            dfs(node, word)
        return result

    def _find_node(self, word: str) -> TrieNode | None:
        node = self.root
        for char in word:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
