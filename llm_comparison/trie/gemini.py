class GeminiCharacterTrie:
    def __init__(self):
        """
        Initializes an empty CharacterTrie.
        Each node is represented by a dictionary:
        - 'children': a dictionary mapping characters to child nodes.
        - 'is_end_of_word': a boolean indicating if this node marks the end of a word.
        """
        self.root = {'children': {}, 'is_end_of_word': False}

    def add(self, word: str) -> None:
        """
        Adds a word into the trie.
        """
        current_node = self.root
        for char in word:
            if char not in current_node['children']:
                current_node['children'][char] = {'children': {}, 'is_end_of_word': False}
            current_node = current_node['children'][char]
        current_node['is_end_of_word'] = True

    def _find_node(self, prefix: str):
        """
        Helper method to find the node corresponding to the end of a given prefix.
        Returns the node if found, otherwise returns None.
        """
        current_node = self.root
        for char in prefix:
            if char not in current_node['children']:
                return None
            current_node = current_node['children'][char]
        return current_node

    def prefix(self, word: str) -> list[str]:
        """
        Returns a list of strings containing all the words in the trie starting with the word,
        including the word itself if it is in the trie.
        """
        results = []
        start_node = self._find_node(word)

        if not start_node:
            return results

        # Helper function for Depth First Search (DFS)
        def _dfs(node, current_prefix):
            if node['is_end_of_word']:
                results.append(current_prefix)
            for char, child_node in node['children'].items():
                _dfs(child_node, current_prefix + char)

        _dfs(start_node, word)
        return results

    def contains(self, word: str) -> bool:
        """
        Returns whether the exact word is in the trie.
        """
        node = self._find_node(word)
        return node is not None and node['is_end_of_word']