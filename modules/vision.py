class Vision():

    def __init__(self, model_name, message, image):
        self.model_name = model_name
        self.message = message
        self.image = image

    def __call__(self):
        prompt = self.message[-1]["content"]
        self.message[-1]["content"] = [
            {
                "type": "image",
                "image": self.image,
            },
            {"type": "text", "text": prompt},
        ]

        from modules.chatbot import ChatBot
        chatbot = ChatBot(self.model_name)
        response = chatbot.send_message_to_model(self.message, {})
        return ChatBot.get_streaming_message(response)
