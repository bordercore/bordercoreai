class Context():

    context_limit = 4096

    def __init__(self, context=None):
        if context is None:
            context = []
        self.context = context

    def add(self, message, prune, role="user", replace_context=False):
        # If we're passing in a list, assume this is the complete
        # chat history and replace it with the new
        if type(message) is list:
            if replace_context:
                self.context = message
            else:
                self.context.extend(message)
        else:
            self.context.append(
                {
                    "role": role,
                    "content": message
                }
            )
        if prune:
            self.prune()

    def get(self):
        return self.context

    def size(self):
        return sum([len(x["role"]) + len(x["content"]) for x in self.context])

    def clear(self):
        self.context = []

    def prune(self):
        # Remove messages from the context, starting at the beginning,
        #  until we're under the context limit.
        # Don't remove the first message, the system message.
        while self.size() > self.context_limit and len(self.context) > 1:
            self.context.pop(1)
