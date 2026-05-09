"""Tests for race_plan_table module."""


class TestRacePlanTable:
    """Test suite for race plan table report generation."""

    def test_generate_race_plan_table_report_creates_html(
        self, sample_course, sample_pacing_df, tmp_path
    ):
        """Test that report generation creates valid HTML file."""
        from race_planner.visualization.race_plan_table import generate_race_plan_table_report

        output_file = tmp_path / "report.html"
        aid_stations = [
            {
                "name": "Start",
                "jap_name": "スタート",
                "distance_km": 0.0,
                "elevation_m": 1000,
                "stop_time_s": 0,
                "notes": "Main start",
            },
            {
                "name": "Finish",
                "jap_name": "ゴール",
                "distance_km": 10.0,
                "elevation_m": 1200,
                "stop_time_s": 300,
                "notes": "Main finish",
            },
        ]

        result = generate_race_plan_table_report(
            course=sample_course,
            aid_stations=aid_stations,
            pacing_df=sample_pacing_df,
            output_path=output_file,
            race_name="Test Race",
            mode="normal",
            race_start_time="06:00:00",
            title="Test Race Plan",
        )

        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "Test Race Plan" in content
        assert "Aid Station Table" in content
        assert "Elevation Profile" in content
        assert 'class="section-meta"' in content
        assert "Planned finish" in content
        assert 'id="target-time-input"' in content
        assert 'id="fatigue-decay-input"' in content
        assert 'id="target-time-apply"' in content
        assert 'id="target-time-reset"' in content
        assert "function applyScale(targetSeconds, newDecayPct)" in content
        assert "function fatigueRatio(progress, newDecayPct)" in content
        assert "function toClockWithDay(totalSeconds)" in content
        assert "clockNode.textContent = toClockWithDay(clockScaled);" in content
        assert 'data-pace-s=' in content
        assert 'class="cell-value emphasized js-pace"' in content
        assert "function toPace(totalSeconds)" in content

    def test_sticky_positioning_in_html(self, sample_course, sample_pacing_df, tmp_path):
        """Test that HTML contains sticky positioning CSS."""
        from race_planner.visualization.race_plan_table import generate_race_plan_table_report

        output_file = tmp_path / "report.html"
        aid_stations = [
            {
                "name": "Start",
                "jap_name": "スタート",
                "distance_km": 0.0,
                "elevation_m": 1000,
                "stop_time_s": 0,
                "notes": "Main start",
            },
            {
                "name": "Finish",
                "jap_name": "ゴール",
                "distance_km": 10.0,
                "elevation_m": 1200,
                "stop_time_s": 300,
                "notes": "Main finish",
            },
        ]

        result = generate_race_plan_table_report(
            course=sample_course,
            aid_stations=aid_stations,
            pacing_df=sample_pacing_df,
            output_path=output_file,
            race_name="Test Race",
            mode="normal",
            race_start_time="06:00:00",
            title="Test Race Plan",
        )

        content = result.read_text(encoding="utf-8")
        assert "position: sticky;" in content
        assert "thead th:first-child" in content
        assert ".sticky-col" in content

    def test_profile_stats_render_in_section_header(
        self, sample_course, sample_pacing_df, tmp_path
    ):
        """Test that profile stats are shown in the section header instead of the plot box."""
        from race_planner.visualization.race_plan_table import generate_race_plan_table_report

        output_file = tmp_path / "report.html"
        aid_stations = [
            {
                "name": "Start",
                "distance_km": 0.0,
                "elevation_m": 1000,
                "stop_time_s": 0,
            },
            {
                "name": "Finish",
                "distance_km": 10.0,
                "elevation_m": 1200,
                "stop_time_s": 300,
            },
        ]

        result = generate_race_plan_table_report(
            course=sample_course,
            aid_stations=aid_stations,
            pacing_df=sample_pacing_df,
            output_path=output_file,
            race_name="Test Race",
            mode="normal",
            title="Test Race Plan",
        )

        content = result.read_text(encoding="utf-8")
        assert 'class="section-meta"' in content
        assert "Total Distance:" not in content

    def test_summary_label_changes_outside_target_time(
        self, sample_course, sample_pacing_df, tmp_path
    ):
        """Non-target-time modes should not label the computed finish as a target."""
        from race_planner.visualization.race_plan_table import generate_race_plan_table_report

        output_file = tmp_path / "report.html"
        aid_stations = [
            {
                "name": "Start",
                "distance_km": 0.0,
                "elevation_m": 1000,
                "stop_time_s": 0,
            },
            {
                "name": "Finish",
                "distance_km": 10.0,
                "elevation_m": 1200,
                "stop_time_s": 300,
            },
        ]

        result = generate_race_plan_table_report(
            course=sample_course,
            aid_stations=aid_stations,
            pacing_df=sample_pacing_df,
            output_path=output_file,
            race_name="Test Race",
            mode="athlete_pb",
            title="Test Race Plan",
        )

        content = result.read_text(encoding="utf-8")
        assert "Planned finish" in content
        assert "Target time" not in content

    def test_generate_report_handles_missing_aid_stations(
        self, sample_course, sample_pacing_df, tmp_path
    ):
        """The report should still render when pacing rows exist without explicit aid station metadata."""
        from race_planner.visualization.race_plan_table import generate_race_plan_table_report

        output_file = tmp_path / "report.html"
        result = generate_race_plan_table_report(
            course=sample_course,
            aid_stations=[],
            pacing_df=sample_pacing_df,
            output_path=output_file,
            race_name="Test Race",
            mode="athlete_pb",
            title="Test Race Plan",
        )

        content = result.read_text(encoding="utf-8")
        assert result.exists()
        assert "Start (スタート)" in content or "Start" in content
        assert "Finish (ゴール)" in content or "Finish" in content

    def test_table_uses_compact_headers_with_split_gain_loss_columns(
        self, sample_course, sample_pacing_df, tmp_path
    ):
        """Table should use compact headers and separate Gain/Loss columns with accum and split lines."""
        from race_planner.visualization.race_plan_table import generate_race_plan_table_report

        output_file = tmp_path / "report.html"
        aid_stations = [
            {
                "name": "Very Long Station Name",
                "distance_km": 0.0,
                "elevation_m": 1000,
                "stop_time_s": 0,
            },
            {
                "name": "Finish",
                "distance_km": 10.0,
                "elevation_m": 1200,
                "stop_time_s": 300,
            },
        ]

        result = generate_race_plan_table_report(
            course=sample_course,
            aid_stations=aid_stations,
            pacing_df=sample_pacing_df,
            output_path=output_file,
            race_name="Test Race",
            mode="target_time",
            title="Test Race Plan",
        )

        content = result.read_text(encoding="utf-8")
        assert "<th>AS</th>" in content
        assert "<th>Dist</th>" in content
        assert "<th>Gain</th>" in content
        assert "<th>Loss</th>" in content
        assert "<th>Time</th>" in content
        assert "<th>Pace</th>" in content
        assert "<th>Notes</th>" in content
        assert "<th>Aid Station</th>" not in content
        assert "<th>Avg Pace / GAP</th>" not in content
        assert ">Σ<" in content
        assert ">Δ<" in content
        assert content.count(">Σ<") == len(sample_pacing_df)
        assert content.count(">Δ<") == len(sample_pacing_df)
        assert "+0 m" in content
        assert "-0 m" in content

    def test_cutoff_time_renders_next_to_clock_line(
        self, sample_course, sample_pacing_df, tmp_path
    ):
        """Cutoff should appear in timing column next to clock time instead of comments."""
        from race_planner.visualization.race_plan_table import generate_race_plan_table_report

        output_file = tmp_path / "report.html"
        aid_stations = [
            {
                "name": "Start",
                "distance_km": 0.0,
                "elevation_m": 1000,
                "stop_time_s": 0,
            },
            {
                "name": "Finish",
                "distance_km": 10.0,
                "elevation_m": 1200,
                "stop_time_s": 300,
                "cutoff_in_time": "D3 16:00",
            },
        ]

        result = generate_race_plan_table_report(
            course=sample_course,
            aid_stations=aid_stations,
            pacing_df=sample_pacing_df,
            output_path=output_file,
            race_name="Test Race",
            mode="target_time",
            title="Test Race Plan",
        )

        content = result.read_text(encoding="utf-8")
        assert "(🚧 D3 16:00)" in content
