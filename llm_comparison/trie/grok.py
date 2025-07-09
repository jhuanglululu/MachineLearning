class GrokCharacterTrie:
    def __init__(self):
        self.root = {}
        self.end_symbol = '*'

    def add(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node[self.end_symbol] = True

    def prefix(self, word: str) -> list[str]:
        node = self.root
        for char in word:
            if char not in node:
                return []
            node = node[char]

        words = []
        self._collect_words(node, word, words)
        return words

    def _collect_words(self, node: dict, prefix: str, words: list[str]) -> None:
        if self.end_symbol in node:
            words.append(prefix)

        for char in node:
            if char != self.end_symbol:
                self._collect_words(node[char], prefix + char, words)

    def contains(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return self.end_symbol in node