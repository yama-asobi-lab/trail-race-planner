from race_planner.main import _build_arg_parser


def test_altitude_effects_cli_default_is_yes():
    parser = _build_arg_parser()
    args = parser.parse_args(["config/races/tgt_2026.yaml"])
    assert args.altitude_effects == "yes"


def test_altitude_effects_cli_can_be_disabled():
    parser = _build_arg_parser()
    args = parser.parse_args(["config/races/tgt_2026.yaml", "--altitude-effects", "no"])
    assert args.altitude_effects == "no"
