class AgentTypes:
    Tank = 0
    MecCar = 1

    SupportsStrafe = {
        Tank: False,
        MecCar: True
    }

    Name = {
        Tank: 'Tank',
        MecCar : 'MecCar'
    }

    PathWidth = {
        Tank: 10,
        MecCar: 10
    }