from position.confidence import Confidence

class AgentActions:
    Go = 0
    Rotate = 2
    GoRandom = 7
    GoToSafePlace = 8
    Heading = 9 # this isn't really an action the vehicle can do, but we need a config for how accurate it should be
    GoForward = 10
    GoReverse = 11
    Strafe = 12
    AdjustRandomly = 13


    # these are trainable actions
    Look = 1
    Photograph = 3
    EstimatePosition = 4
    Nothing = 5
    ReportFound = 6

    # these are trainable actions that compose the other actions
    # above. their accuracy is determined by the methods they utilize
    GoForwardShort = 20
    GoForwardMedium = 21
    GoForwardFar = 22
    GoReverseShort = 23
    GoReverseMedium = 24
    GoReverseFar = 25
    RotateLeftSmall = 26
    RotateLeftMedium = 27
    RotateLeftBig = 28
    RotateRightSmall = 26
    RotateRightMedium = 27
    RotateRightBig = 28

    #GoFullDistance = 10 # this isn't really an action the vehicle can do, but we need a config for how likely vehicle is to stop short

    Accuracy = {
        Go : 0.97, 
        GoForward : 0.97,
        GoReverse : 0.97,
        Strafe : 0.97,
        Look : 0.9, # how often the target is found, when the within vision range
        Rotate : 0.98,
        Photograph : 0.8, # how often the photograph is successful if within photograhp range
        EstimatePosition : {
            Confidence.CONFIDENCE_HIGH: 0.95,
            Confidence.CONFIDENCE_MEDIUM: 0.9,
        },
        Heading : {
            Confidence.CONFIDENCE_HIGH: 0.98,
            Confidence.CONFIDENCE_MEDIUM: 0.96,
        },
        Nothing : 1,
        ReportFound : 0.8,
        GoRandom : 0.96, # accurcy here has an oversized effect, since the coords being adjusted are mesa coords, not lvps
        GoToSafePlace : 0.96, # accurcy here has an oversized effect, since the coords being adjusted are mesa coords, not lvps
        AdjustRandomly : 0.95, # no effect really
        Strafe : 0.95
    }

    SuccessRate = {
        Go : 0.8,
        GoForward : 0.95,
        GoReverse : 0.95,
        Strafe : 0.95,
        Look : 0.9,
        Rotate : 0.97,
        Photograph : 0.9,
        EstimatePosition : 0.75,
        Nothing : 1,
        ReportFound : 0.9,
        GoRandom : 0.95,
        GoToSafePlace : 0.95,
        AdjustRandomly : 0.95,
        Strafe : 0.95
        #GoFullDistance : .8 # percent of times it goes the full distance
    }

    StepCost = {
        Go : 2,
        Look : 2,
        Rotate : 1,
        Photograph : 2,
        EstimatePosition : 3,
        Nothing : 1,
        ReportFound : 1,
        GoRandom : 1,
        GoToSafePlace : 2,
        GoForward : 1,
        GoReverse : 1,
        AdjustRandomly : 1,
        Strafe : 1,

        GoForwardShort : 1,
        GoForwardMedium : 1,
        GoForwardFar : 1,
        GoReverseShort : 1,
        GoReverseMedium : 1,
        GoReverseFar : 1,
        RotateLeftSmall : 1,
        RotateLeftMedium : 1,
        RotateLeftBig : 1,
        RotateRightSmall : 1,
        RotateRightMedium : 1,
        RotateRightBig : 1
    }