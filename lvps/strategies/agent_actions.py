from position.confidence import Confidence

class AgentActions:
    NumTrainableActions = 14

    # these are trainable actions. They need to be a sequence starting with zero for gym environment to use them
    Look = 0
    Photograph = 1

    # these are trainable actions that compose the other actions
    # above. their accuracy is determined by the methods they utilize
    GoForwardShort = 2
    GoForwardMedium = 3
    GoForwardFar = 4
    GoReverseShort = 5
    GoReverseMedium = 6
    GoReverseFar = 7
    RotateLeftSmall = 8
    RotateLeftMedium = 9
    RotateLeftBig = 10
    RotateRightSmall = 11
    RotateRightMedium = 12
    RotateRightBig = 13

    Nothing = 14
    ReportFound = 15

    EstimatePosition = 16
    Go = 17
    Rotate = 18
    GoRandom = 19
    GoToSafePlace = 20
    Heading = 21 # this isn't really an action the vehicle can do, but we need a config for how accurate it should be
    GoForward = 22
    GoReverse = 23
    Strafe = 24
    AdjustRandomly = 25

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
        Go : 1,
        Look : 2,
        Rotate : 1,
        Photograph : 2,
        EstimatePosition : 1,
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

    Names = {
        Look : "Look",
        Photograph : "Photograph",
        EstimatePosition : "EstimatePosition",
        Nothing : "Nothing",
        ReportFound : "ReportFound",
        GoForwardShort : "GoForwardShort",
        GoForwardMedium : "GoForwardMedium",
        GoForwardFar : "GoForwardFar",
        GoReverseShort : "GoReverseShort",
        GoReverseMedium : "GoReverseMedium",
        GoReverseFar : "GoReverseFar",
        RotateLeftSmall : "RotateLeftSmall",
        RotateLeftMedium : "RotateLeftMedium",
        RotateLeftBig : "RotateLeftBig",
        RotateRightSmall : "RotateRightSmall",
        RotateRightMedium : "RotateRightMedium",
        RotateRightBig : "RotateRightBig",

        Go : "Go",
        Rotate : "Rotate",
        GoRandom : "GoRandom",
        GoToSafePlace : "GoToSafePlace",
        Heading : "Heading",
        GoForward : "GoForward",
        GoReverse : "GoReverse",
        Strafe : "Strafe",
        AdjustRandomly : "AdjustRandomly"
    }