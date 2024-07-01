import json
from datetime import datetime, timedelta

import dateutil.parser
import httplib2
from api import settings
from apiclient.discovery import build
from oauth2client.client import OAuth2Credentials

from .rfc3339 import datetimetostr
from .rfc3339 import now as now_rfc3339


def get_schedule(model_name, command):

    try:
        with open("./calendar_credentials.json", "r") as file:
            cal_info = json.load(file)
    except FileNotFoundError:
        raise Exception("Calendar credentials not found.")

    credentials = OAuth2Credentials(
        cal_info["access_token"],
        cal_info["client_id"],
        cal_info["client_secret"],
        cal_info["refresh_token"],
        cal_info["token_expiry"],
        cal_info["token_uri"],
        cal_info["user_agent"],
        cal_info["revoke_uri"],
        cal_info["id_token"],
        cal_info["token_response"],
    )

    http = httplib2.Http()
    http = credentials.authorize(http)
    service = build(
        serviceName="calendar",
        version="v3",
        http=http,
        developerKey="",
        cache_discovery=False
    )
    timeMax = datetime.now() + timedelta(days=7)

    events = service.events().list(calendarId=settings.calendar_id,
                                   orderBy="startTime",
                                   singleEvents=True,
                                   timeMin=str(now_rfc3339()).replace(" ", "T"),
                                   timeMax=datetimetostr(timeMax)).execute()

    prompt = f"""
    I will give you a series of events on my personal calendar. I want you to answer a question about my calendar based on those events. For reference assume today is {datetime.now().strftime("%A")}. The question is the following: {command}. Answer that question based on the following list of calendar events.
    """

    for event in events["items"]:
        start_pretty = dateutil.parser.parse(event["start"]["dateTime"]).strftime("%a %I:%M%p")
        prompt = prompt + f"""
        One event is called {event['summary']} and starts on {start_pretty}
        """

    print(prompt)
    args = {"temperature": 0.1}

    from chatbot import ChatBot
    chatbot = ChatBot(model_name)
    response = chatbot.send_message_to_model(prompt, args)

    return {
        "content": response["content"],
        "speed": response["speed"]
    }
