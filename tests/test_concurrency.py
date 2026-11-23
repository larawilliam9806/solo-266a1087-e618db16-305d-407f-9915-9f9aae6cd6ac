"""Concurrency guarantees between long reviews and backfill uploads.

A review over a message-heavy window must not lock out same-station backfills,
the review's two reads must share one fixed snapshot, a review only becomes
visible after its full write commits, and a failed review write leaves an
already effective backfill in place.
"""

import threading

from conftest import observation_csv, load, make_review, results_for
from app import db, ingest, review

T = 180.0
START = "2026-05-01 00:00"
END = "2026-05-01 12:00"


def _seed(conn):
    load(
        conn,
        "S1",
        forecasts=[
            ("2026-05-01 05:00", "2026-04-30 10:00", "2026-04-30 10:10", 185),
        ],
        observations=[
            ("2026-05-01 04:00", 120),
            ("2026-05-01 05:00", 200),
        ],
    )


def _backfill_removing_crossing(db_path):
    """Correction through a *fresh* connection: 04:00 moves above the line."""
    conn = db.connect(db_path)
    try:
        batch = ingest.parse_batch(
            station="S1",
            forecast_bytes=None,
            observation_bytes=observation_csv([("2026-05-01 04:00", 195)]),
        )
        ingest.commit_batch(conn, batch, observation_name="correction.csv")
    finally:
        conn.close()


def test_backfill_commits_while_review_is_being_computed(app, conn):
    """A backfill's BEGIN IMMEDIATE must not wait for the review's read phase."""
    _seed(conn)
    review_started = threading.Event()
    release_review = threading.Event()

    original_find_crossings = review.find_crossings

    def slow_find_crossings(*args, **kwargs):
        # We are inside the review's read transaction, just after the
        # observation read that drives crossing detection.
        review_started.set()
        assert release_review.wait(timeout=10), "review never paused"
        return original_find_crossings(*args, **kwargs)

    review.find_crossings = slow_find_crossings
    try:
        outcome = {}

        def run_review():
            with app.app_context():
                review_conn = app.get_conn()
                try:
                    rid = make_review(review_conn, "S1", START, END, T)
                    outcome["rid"] = rid
                except Exception as exc:  # pragma: no cover - surfaced below
                    outcome["error"] = exc

        worker = threading.Thread(target=run_review)
        worker.start()
        assert review_started.wait(timeout=5), "review did not reach read phase"

        # While the review still holds its (read-only) snapshot, the backfill
        # must commit through the original entry point without timing out.
        committed = threading.Event()

        def run_backfill():
            _backfill_removing_crossing(app.config["DATABASE"])
            committed.set()

        filler = threading.Thread(target=run_backfill)
        filler.start()
        assert committed.wait(timeout=5), "backfill blocked behind the review"
        filler.join(timeout=5)

        release_review.set()
        worker.join(timeout=10)
    finally:
        review.find_crossings = original_find_crossings

    assert "error" not in outcome, outcome.get("error")
    rid = outcome["rid"]

    # The review froze the data as of its first read: the crossing at 05:00
    # recorded from the old observations, even though the backfill committed
    # before the review's own forecast read and write.
    rows = results_for(conn, rid)
    assert list(rows) == ["2026-05-01T05:00:00Z"]
    row = rows["2026-05-01T05:00:00Z"]
    assert row["status"] == "crossing"
    assert row["prev_level_cm"] == 120
    # Forecasts and observations in the snapshot must come from that same
    # pre-backfill moment.
    snap_obs = {
        r["hour_utc"]: r["level_cm"]
        for r in conn.execute(
            "SELECT hour_utc, level_cm FROM review_observations WHERE review_id = ?", (rid,)
        ).fetchall()
    }
    assert snap_obs["2026-05-01T04:00:00Z"] == 120

    # A review started only after the backfill uses the corrected data: with
    # 04:00 above the line the 05:00 crossing is gone (04:00 itself is only
    # 待核 because its own previous hour is missing).
    rid2 = make_review(conn, "S1", START, END, T)
    rows2 = results_for(conn, rid2)
    assert "2026-05-01T05:00:00Z" not in rows2
    assert rows2["2026-05-01T04:00:00Z"]["status"] == "pending"

    # The correction itself stays effective on the main tables.
    corrected = conn.execute(
        "SELECT level_cm FROM observation WHERE station = 'S1' AND hour_utc = ?",
        ("2026-05-01T04:00:00Z",),
    ).fetchone()
    assert corrected["level_cm"] == 195


def test_backfill_through_http_succeeds_during_inflight_review(app, client, conn):
    """Same guarantee via the /upload entry point a duty officer actually uses."""
    _seed(conn)
    review_started = threading.Event()
    release_review = threading.Event()
    original_find_crossings = review.find_crossings

    def slow_find_crossings(*args, **kwargs):
        review_started.set()
        assert release_review.wait(timeout=10)
        return original_find_crossings(*args, **kwargs)

    review.find_crossings = slow_find_crossings
    try:
        outcome = {}

        def run_review():
            with app.app_context():
                try:
                    outcome["rid"] = make_review(app.get_conn(), "S1", START, END, T)
                except Exception as exc:  # pragma: no cover
                    outcome["error"] = exc

        worker = threading.Thread(target=run_review)
        worker.start()
        assert review_started.wait(timeout=5)

        import io

        resp_holder = {}

        def run_upload():
            resp_holder["resp"] = client.post(
                "/upload",
                data={
                    "station": "S1",
                    "observation_file": (
                        io.BytesIO(observation_csv([("2026-05-01 04:00", 195)])),
                        "correction.csv",
                    ),
                },
                content_type="multipart/form-data",
            )

        uploader = threading.Thread(target=run_upload)
        uploader.start()
        uploader.join(timeout=10)
        assert not uploader.is_alive(), "upload timed out while review was running"
        release_review.set()
        worker.join(timeout=10)
    finally:
        review.find_crossings = original_find_crossings

    # Under the old single-write-transaction design this would have been a 500
    # "database is locked"; now the backfill commits normally.
    assert resp_holder["resp"].status_code == 302
    assert "error" not in outcome
    assert conn.execute(
        "SELECT level_cm FROM observation WHERE station = 'S1' AND hour_utc = ?",
        ("2026-05-01T04:00:00Z",),
    ).fetchone()["level_cm"] == 195


def test_failed_review_write_is_hidden_and_backfill_stays_effective(app, client, conn):
    """If the write phase fails, no partial review shows; the backfill stands."""
    _seed(conn)
    # A backfill committed before the review must survive the later failure.
    _backfill_removing_crossing(app.config["DATABASE"])

    def break_writes(_conn):
        # Separate connection: drops happen as their own committed transaction,
        # then the review's open write transaction fails on its first insert.
        side = db.connect(app.config["DATABASE"])
        try:
            side.execute("DROP TABLE review_results")
        finally:
            side.close()

    review._pre_write_hook = break_writes
    try:
        resp = client.post(
            "/reviews",
            data={
                "station": "S1",
                "period_start": "2026-05-01T00:00",
                "period_end": "2026-05-01T12:00",
                "threshold_cm": str(T),
            },
            follow_redirects=True,
        )
    finally:
        review._pre_write_hook = None

    assert resp.status_code == 200
    text = resp.get_data(as_text=True)
    assert "回查结果保存失败" in text

    # Nothing of the failed review may be visible anywhere.
    assert conn.execute("SELECT COUNT(*) c FROM reviews").fetchone()["c"] == 0
    assert conn.execute("SELECT COUNT(*) c FROM review_observations").fetchone()["c"] == 0
    assert conn.execute("SELECT COUNT(*) c FROM review_forecasts").fetchone()["c"] == 0
    listing = client.get("/reviews").get_data(as_text=True)
    assert "还没有回查记录" in listing

    # The previously committed backfill is untouched and still effective.
    assert conn.execute(
        "SELECT level_cm FROM observation WHERE station = 'S1' AND hour_utc = ?",
        ("2026-05-01T04:00:00Z",),
    ).fetchone()["level_cm"] == 195


def test_review_listed_only_after_snapshot_and_results_commit(app, client, conn):
    """Detail and CSV of a committed review read exactly that review's data."""
    _seed(conn)
    seen = {}

    def observe(_conn):
        # From another connection the review row must not exist yet while its
        # snapshot and results are still being written.
        side = db.connect(app.config["DATABASE"])
        try:
            seen["reviews_during_write"] = side.execute(
                "SELECT COUNT(*) c FROM reviews"
            ).fetchone()["c"]
        finally:
            side.close()

    review._pre_write_hook = observe
    try:
        rid = make_review(conn, "S1", START, END, T)
    finally:
        review._pre_write_hook = None

    assert seen["reviews_during_write"] == 0

    # Only after the commit does the review appear, and detail/export read the
    # stored snapshot rather than the live tables.
    listing = client.get("/reviews").get_data(as_text=True)
    assert str(rid) in listing
    detail = client.get(f"/reviews/{rid}")
    assert detail.status_code == 200
    assert "2026-05-01 05:00Z" in detail.get_data(as_text=True)
    export = client.get(f"/reviews/{rid}/export.csv")
    assert export.status_code == 200
    assert "2026-05-01 05:00Z" in export.get_data(as_text=True)
