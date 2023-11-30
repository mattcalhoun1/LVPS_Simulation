class SearchAgentActions:
    Go = 0
    Look = 1
    Rotate = 2
    Photograph = 3
    EstimatePosition = 4
    Nothing = 5
    ReportFound = 6
    GoRandom = 7
    GoToSafePlace = 8

    Accuracy = {
        Go : 0.85,
        Look : 0.8, # how often the target is found, when the within vision range
        Rotate : 1.0,
        Photograph : 0.8, # how often the photograph is successful if within photograhp range
        EstimatePosition : 0.94,
        Nothing : 1,
        ReportFound : 0.8,
        GoRandom : 0.96, # accurcy here has an oversized effect, since the coords being adjusted are mesa coords, not lvps
        GoToSafePlace : 0.96 # accurcy here has an oversized effect, since the coords being adjusted are mesa coords, not lvps
    }

    SuccessRate = {
        Go : 0.95,
        Look : 0.95,
        Rotate : 0.95,
        Photograph : 0.8,
        EstimatePosition : 0.75,
        Nothing : 1,
        ReportFound : 0.9,
        GoRandom : 0.95,
        GoToSafePlace : 0.95
    }

    StepCost = {
        Go : 1,
        Look : 2,
        Rotate : 1,
        Photograph : 2,
        EstimatePosition : 2,
        Nothing : 1,
        ReportFound : 1,
        GoRandom : 1,
        GoToSafePlace : 2
    }
