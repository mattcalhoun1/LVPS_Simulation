class SimEventType:
    AgentMoved = 0
    AgentLooked = 1
    AgentRotated = 2
    TargetFound = 3

class SimEventSubscriptions:
    def __init__(self):
        self.__subscriptions = {}

    def add_subscription (self, event_type, listener):
        if event_type not in [SimEventType.AgentMoved, SimEventType.AgentLooked, SimEventType.AgentRotated, SimEventType.TargetFound]:
            raise Exception("Invalid event")

        if event_type not in self.__subscriptions:
            self.__subscriptions[event_type] = []
        self.__subscriptions[event_type].append(listener)

    def notify_subscribers (self, event_type, event_details):
        if event_type in self.__subscriptions:
            for s in self.__subscriptions[event_type]:
                s.handle_event(event_type, event_details)