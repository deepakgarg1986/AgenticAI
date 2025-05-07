def chat_styles():
    return """
    <style>
        .user-message {
            background-color: #E0E0E0;
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
            width: 90%;
            text-align: left;
        }
        .assistant-message {
            background-color: #E0E0E0;
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
            width: 90%;
            text-align: left;
            float: right;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
    </style>
    """
