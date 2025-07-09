class DeepseekCharacterTrie:
    class TrieNode:
        __slots__ = ('children', 'is_end')
        def __init__(self):
            self.children = {}
            self.is_end = False

    def __init__(self):
        self.root = self.TrieNode()

    def add(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = self.TrieNode()
            node = node.children[char]
        node.is_end = True

    def contains(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def prefix(self, prefix: str) -> list[str]:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        stack = [(node, prefix)]
        words = []
        while stack:
            current, current_string = stack.pop()
            if current.is_end:
                words.append(current_string)
            keys = sorted(current.children.keys(), reverse=True)
            for char in keys:
                child = current.children[char]
                stack.append((child, current_string + char))
        return words