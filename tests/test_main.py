from main import build_parser


def test_no_args_means_interactive_mode():
    args = build_parser().parse_args([])
    assert args.command is None


def test_build_index_subcommand():
    args = build_parser().parse_args(["build-index"])
    assert args.command == "build-index"


def test_dashboard_subcommand_with_no_flags():
    args = build_parser().parse_args(["dashboard"])
    assert args.command == "dashboard"
    assert args.tasks is None
    assert args.compare is None


def test_dashboard_subcommand_with_tasks_flag():
    args = build_parser().parse_args(["dashboard", "--tasks", "184,185,190"])
    assert args.tasks == "184,185,190"


def test_dashboard_subcommand_with_compare_flag():
    args = build_parser().parse_args(["dashboard", "--compare", "184", "185"])
    assert args.compare == ["184", "185"]


def test_timeline_subcommand_requires_task_id():
    args = build_parser().parse_args(["timeline", "184"])
    assert args.command == "timeline"
    assert args.task_id == "184"
