if [ -z "$1" ]; then
    echo "Error: No chatbot argument supplied. Must be 'chatgpt' or 'floyd'" >&2
    exit 1
fi

source /home/jerrell/.bashrc-bordercore
/home/jerrell/dev/envs/chatbot/bin/python3 /home/jerrell/dev/chatbot/chatbot.py -m $1
