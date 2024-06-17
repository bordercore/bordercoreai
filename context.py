class Context():

    context_limit = 4096

    def __init__(self):
        self.context = []

    def add(self, role, message):
        self.context.append(
            {
                "role": role,
                "content": message
            }
        )
        self.prune()

    def get(self):
        return self.context

    def set(self, context):
        self.context = context

    def size(self):
        return len(self.get())

    def clear(self):
        self.context = []

    def prune(self):
        for message in list(self.context):
            if self.size() < self.context_limit:
                break
            self.context.pop(0)
