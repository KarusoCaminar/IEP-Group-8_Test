from dataclasses import dataclass


@dataclass(frozen=True)
class Port:
    unit_name: str
    port: str


@dataclass(frozen=True)
class Connector:
    name: str
    to_converter_ports: tuple[Port, ...]
    from_converter_ports: tuple[Port, ...]


@dataclass(frozen=True)
class Network:
    connectors: tuple[Connector, ...]


cgm_network = Network(
    connectors=(
        Connector(
            name="Node 1",
            from_converter_ports=(Port(unit_name="Consumer", port="medium"),),
            to_converter_ports=(Port(unit_name="Primary Pump", port="medium"),),
        ),
        Connector(
            name="Node 2",
            from_converter_ports=(Port(unit_name="Primary Pump", port="medium"),),
            to_converter_ports=(
                Port(unit_name="Cooling Tower", port="cooling"),
                Port(unit_name="Valve", port="hot"),
            ),
        ),
        Connector(
            name="Node 3",
            from_converter_ports=(Port(unit_name="Cooling Tower", port="cooling"),),
            to_converter_ports=(Port(unit_name="Valve", port="medium"),),
        ),
        Connector(
            name="Node 4",
            from_converter_ports=(Port(unit_name="Valve", port="medium"),),
            to_converter_ports=(
                Port(unit_name="Bypass", port="medium"),
                Port(unit_name="Chiller", port="cooling"),
            ),
        ),
        Connector(
            name="Node 5",
            from_converter_ports=(
                Port(unit_name="Bypass", port="medium"),
                Port(unit_name="Chiller", port="cooling"),
            ),
            to_converter_ports=(Port(unit_name="Consumer", port="medium"),),
        ),
    )
)
