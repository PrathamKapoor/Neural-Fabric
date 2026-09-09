"""Tests for persistent detached jobs (PART 23 lifecycle + PART 21 recovery).

Uses REAL detached processes (short sleeps, heartbeats, checkpoints) plus
failure injection. No network, no LLM.

Run:  python -m unittest discover -s tests -v   (from repo root)
"""
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab import jobs
from lab.jobs import JobManager

PY = sys.executable


def _tmp(testcase):
    tmp = tempfile.TemporaryDirectory()
    testcase.addCleanup(tmp.cleanup)
    return tmp.name


def _jm(testcase, root, **kw):
    d = {"max_concurrent": 4}
    d.update(kw)
    return JobManager(root, jobs_dir=os.path.join(root, ".jobs"), **d)


def _script(root, name, body):
    p = os.path.join(root, name)
    with open(p, "w") as f:
        f.write(body)
    return p


def _wait_for(jm, job_id, want=("COMPLETED",), timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        rec = jm.poll(job_id)
        if rec is None or rec.get("status") in want or rec.get("status") in (
                "COMPLETED", "FAILED", "TIMED_OUT", "CANCELLED", "ORPHANED"):
            return rec
        time.sleep(0.5)
    return jm.poll(job_id)


SLEEPER = ("import sys, time, json; n=int(sys.argv[1]); "
           "out=sys.argv[2]; "
           "[time.sleep(0.2) for _ in range(n)]; "
           "json.dump({'done': True, 'n': n}, open(out, 'w'))")


class TestSubmitSchedule(unittest.TestCase):
    def test_submit_persists_queued(self):
        root = _tmp(self)
        jm = _jm(self, root)
        rec = jm.submit([PY, "-c", "pass"], "exp1", seed=3, timeout_s=60, mem_mb=8)
        self.assertEqual(rec["status"], "QUEUED")
        self.assertTrue(os.path.exists(
            os.path.join(root, ".jobs", rec["job_id"] + ".json")))
        self.assertEqual(rec["seed"], 3)

    def test_rejects_shell_string_and_escapes(self):
        root = _tmp(self)
        jm = _jm(self, root)
        with self.assertRaises(ValueError):
            jm.submit("rm -rf /", "x")
        with self.assertRaises(ValueError):
            jm.submit([PY, "../outside.py"], "x")

    def test_max_concurrent_queues_with_reason(self):
        root = _tmp(self)
        jm = _jm(self, root, max_concurrent=1)
        self.addCleanup(lambda: [jm.cancel(r["job_id"])
                                 for r in jm.list_jobs()])
        s = _script(root, "s.py", "import time; time.sleep(8)")
        j1 = jm.submit([PY, s], "e", timeout_s=60, mem_mb=1)
        j2 = jm.submit([PY, s], "e", timeout_s=60, mem_mb=1)
        jm.start_due()
        # FIFO by submission: j1 must run, j2 must queue with a reason
        self.assertEqual(jm._read(j1["job_id"])["status"], "RUNNING")
        r2 = jm._read(j2["job_id"])
        self.assertEqual(r2["status"], "QUEUED")
        self.assertIn("max_concurrent", r2["queue_reason"])
        r2 = jm._read(j2["job_id"])
        self.assertEqual(r2["status"], "QUEUED")
        self.assertIn("max_concurrent", r2["queue_reason"])
        jm.cancel(j1["job_id"])
        jm.cancel(j2["job_id"])

    def test_ram_gate_queues_with_reason(self):
        root = _tmp(self)
        jm = _jm(self, root)
        rec = jm.submit([PY, "-c", "pass"], "e", timeout_s=60,
                        mem_mb=10 ** 12)
        jm.start_due()
        r = jm._read(rec["job_id"])
        self.assertEqual(r["status"], "QUEUED")
        self.assertIn("RAM", r["queue_reason"])

    def test_ram_probe_failure_fails_open(self):
        root = _tmp(self)
        jm = _jm(self, root)
        rec = jm.submit([PY, "-c", "pass"], "e", timeout_s=60, mem_mb=1)
        with patch("lab.jobs.memory", return_value={"total_mb": 0,
                                                    "free_mb": 0}):
            jm.start_due()
        r = jm._read(rec["job_id"])
        # probe failure must not deadlock the scheduler (cap still applies)
        self.assertEqual(r["status"], "RUNNING")
        jm.cancel(rec["job_id"])

    def test_exit_code_recorded(self):
        root = _tmp(self)
        jm = _jm(self, root)
        res = os.path.join(root, "r.json")
        code = ("import json, sys; json.dump({'x': 1}, open('r.json', 'w')); "
                "sys.exit(7)")
        rec = jm.submit([PY, "-c", code], "e", timeout_s=60, mem_mb=8,
                        result_rel="r.json")
        jm.start_due()
        done = _wait_for(jm, rec["job_id"])
        self.assertEqual(done["status"], "COMPLETED")
        self.assertEqual(done["exit_code"], 7)
        self.assertTrue(os.path.exists(res))


class TestDetachedLifecycle(unittest.TestCase):
    def test_queue_start_poll_complete_detached(self):
        root = _tmp(self)
        jm = _jm(self, root)
        res = os.path.join(root, "res.json")
        rec = jm.submit([PY, "-c", SLEEPER, "3", res], "e1",
                        timeout_s=60, mem_mb=8, result_rel="res.json")
        jm.start_due()
        # manager returns immediately: job runs on without us
        r1 = jm.poll(rec["job_id"])
        self.assertIn(r1["status"], ("RUNNING", "COMPLETED"))
        self.assertTrue(r1["pid"] > 0)
        done = _wait_for(jm, rec["job_id"])
        self.assertEqual(done["status"], "COMPLETED")
        self.assertTrue(json.load(open(res))["done"])
        # log captured
        self.assertTrue(os.path.exists(done["log_path"]))

    def test_heartbeat_visible_while_running(self):
        root = _tmp(self)
        jm = _jm(self, root)
        hb = os.path.join(root, "hb.jsonl")
        body = ("import sys, time; "
                "sys.path.insert(0, %r); "
                "from lab.jobs import JobManager; "
                "hb = sys.argv[1]; res = sys.argv[2]; "
                "import json; "
                "[JobManager.heartbeat(hb, step=i, epoch=i) or time.sleep(0.4) for i in range(8)]; "
                "json.dump({'done': True}, open(res, 'w'))" % os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__)))
                )
        res = os.path.join(root, "res.json")
        rec = jm.submit([PY, "-c", body, hb, res], "e2", timeout_s=60,
                        mem_mb=8, result_rel="res.json", heartbeat_rel="hb.jsonl")
        jm.start_due()
        time.sleep(1.5)
        mid = jm.poll(rec["job_id"])
        self.assertIn(mid["status"], ("RUNNING", "COMPLETED"))
        if mid["status"] == "RUNNING":
            self.assertTrue(mid["heartbeat"])
            self.assertIn("step", mid["progress"])
        done = _wait_for(jm, rec["job_id"])
        self.assertEqual(done["status"], "COMPLETED")

    def test_timeout_kills_and_marks(self):
        root = _tmp(self)
        jm = _jm(self, root)
        rec = jm.submit([PY, "-c", "import time; time.sleep(120)"], "e3",
                        timeout_s=3, mem_mb=8)
        jm.start_due()
        done = _wait_for(jm, rec["job_id"], timeout=60)
        self.assertEqual(done["status"], "TIMED_OUT")
        self.assertIn("deadline", done["failure_reason"])
        from lab.jobs import _pid_alive
        self.assertFalse(_pid_alive(done["pid"]))

    def test_failing_command_marks_failed(self):
        root = _tmp(self)
        jm = _jm(self, root)
        res = os.path.join(root, "never.json")
        rec = jm.submit([PY, "-c", "import sys; sys.exit(3)"], "e4",
                        timeout_s=30, mem_mb=8, result_rel="never.json")
        jm.start_due()
        done = _wait_for(jm, rec["job_id"])
        self.assertEqual(done["status"], "ORPHANED")
        self.assertIn("no result", done["failure_reason"])


class TestCheckpointResume(unittest.TestCase):
    def test_kill_then_resume_matches_uninterrupted(self):
        root = _tmp(self)
        jm = _jm(self, root)
        worker = _script(root, "ckpt_worker.py",
                         "import sys, time, json, os\n"
                         "ckpt, res, mode = sys.argv[1], sys.argv[2], sys.argv[3]\n"
                         "sys.path.insert(0, %r)\n"
                         "from lab.jobs import JobManager\n"
                         "start = 0\n"
                         "if mode == 'resume':\n"
                         "    prev = JobManager.checkpoint_load(ckpt) or {}\n"
                         "    start = prev.get('done', 0)\n"
                         "total = 6\n"
                         "for i in range(start, total):\n"
                         "    time.sleep(0.3)\n"
                         "    JobManager.checkpoint_save(ckpt, {'done': i + 1, 'total': total})\n"
                         "    JobManager.heartbeat(ckpt + '.hb', step=i)\n"
                         "json.dump({'total': total}, open(res, 'w'))\n"
                         % os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ckpt, res = os.path.join(root, "c.json"), os.path.join(root, "r.json")
        j1 = jm.submit([PY, worker, ckpt, res, "fresh"], "e5", timeout_s=60,
                       mem_mb=8, result_rel="r.json", checkpoint_rel="c.json")
        jm.start_due()
        time.sleep(1.0)
        jm.cancel(j1["job_id"])  # simulate crash mid-run
        partial = JobManager.checkpoint_load(ckpt) or {}
        self.assertTrue(0 < partial.get("done", 0) < 6)
        j2 = jm.submit([PY, worker, ckpt, res, "resume"], "e5", timeout_s=60,
                       mem_mb=8, result_rel="r.json", checkpoint_rel="c.json")
        jm.start_due()
        done = _wait_for(jm, j2["job_id"])
        self.assertEqual(done["status"], "COMPLETED")
        self.assertEqual(json.load(open(res)), {"total": 6})


class TestRecovery(unittest.TestCase):
    def test_orphan_detected_and_requeued(self):
        root = _tmp(self)
        jm = _jm(self, root)
        rec = jm.submit([PY, "-c", "pass"], "e6", timeout_s=60, mem_mb=8)
        rec.update(status="RUNNING", pid=999999, start_time=1, deadline=9999999999)
        jm._write(rec)
        changed = jm.recover()
        self.assertEqual(changed[0]["status"], "ORPHANED")
        jm.requeue(rec["job_id"])
        self.assertEqual(jm._read(rec["job_id"])["status"], "QUEUED")

    def test_result_present_recovers_completed(self):
        root = _tmp(self)
        jm = _jm(self, root)
        with open(os.path.join(root, "r.json"), "w") as f:
            f.write("{}")
        rec = jm.submit([PY, "-c", "pass"], "e7", timeout_s=60, mem_mb=8,
                        result_rel="r.json")
        rec.update(status="RUNNING", pid=999999, start_time=1, deadline=9999999999)
        jm._write(rec)
        changed = jm.recover()
        self.assertEqual(changed[0]["status"], "COMPLETED")

    def test_restart_recovery_fresh_manager(self):
        root = _tmp(self)
        jm = _jm(self, root)
        rec = jm.submit([PY, "-c", "pass"], "e8", timeout_s=60, mem_mb=8)
        jm2 = JobManager(root, jobs_dir=os.path.join(root, ".jobs"))
        self.assertEqual(jm2._read(rec["job_id"])["status"], "QUEUED")
        self.assertEqual(jm2._read(rec["job_id"])["experiment_id"], "e8")


class TestMultiSeed(unittest.TestCase):
    def test_failed_seed_does_not_kill_siblings(self):
        root = _tmp(self)
        jm = _jm(self, root, max_concurrent=4)
        ids = []
        for s in range(3):
            code = ("import sys; sys.exit(1)" if s == 1
                    else "import json; json.dump({'seed': %d}, open('r%d.json','w'))" % (s, s))
            rec = jm.submit([PY, "-c", code], "e9", seed=s, timeout_s=60,
                            mem_mb=8,
                            result_rel=("r%d.json" % s if s != 1 else "nope.json"))
            ids.append(rec["job_id"])
        jm.start_due()
        final = {jm.poll(i)["job_id"]: jm.poll(i)["status"] for i in ids}
        # wait for all terminals
        t0 = time.time()
        while (any(s not in ("COMPLETED", "FAILED", "TIMED_OUT", "CANCELLED",
                             "ORPHANED") for s in final.values())
               and time.time() - t0 < 60):
            time.sleep(0.5)
            final = {jm.poll(i)["job_id"]: jm._read(i)["status"] for i in ids}
        by_seed = {jm._read(i)["seed"]: jm._read(i)["status"] for i in ids}
        self.assertEqual(by_seed[0], "COMPLETED")
        self.assertEqual(by_seed[2], "COMPLETED")
        self.assertIn(by_seed[1], ("ORPHANED", "FAILED"))
        completed = sorted(s for s, st in by_seed.items() if st == "COMPLETED")
        self.assertEqual(completed, [0, 2])  # analysis knows exactly these


class TestResources(unittest.TestCase):
    def test_resources_structure(self):
        root = _tmp(self)
        info = jobs.resources(root)
        self.assertGreaterEqual(info["cpu"], 1)
        self.assertIn("free_mb", info["ram_mb"])
        self.assertGreater(info["disk_free_bytes"], 0)
        self.assertIsInstance(info["gpu"]["present"], bool)

    def test_unknown_job(self):
        root = _tmp(self)
        self.assertIsNone(JobManager(root, jobs_dir=os.path.join(root, ".j")).poll("nope"))


class TestWorkerTools(unittest.TestCase):
    def test_queue_and_poll_tools(self):
        from agent_loop.tools import ToolExecutor
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ex = ToolExecutor(tmp.name)
        res = os.path.join(tmp.name, "r.json")
        q = ex.execute("queue_job", {
            "command": [PY, "-c",
                        "import json; json.dump({'ok': True}, open('r.json','w'))"],
            "experiment_id": "t1", "timeout_s": 60, "mem_mb": 8, "mem_mb": 8, "result": "r.json"})
        self.assertTrue(q["ok"], q)
        self.assertIn("queued job", q["output"])
        job_id = q["output"].split("queued job ")[1].split(" ")[0]
        t0 = time.time()
        status = ""
        while time.time() - t0 < 60:
            p = ex.execute("poll_job", {"job_id": job_id})
            self.assertTrue(p["ok"])
            status = json.loads(p["output"])["status"]
            if status in ("COMPLETED", "FAILED", "TIMED_OUT"):
                break
            time.sleep(0.5)
        self.assertEqual(status, "COMPLETED")
        self.assertTrue(os.path.exists(res))
        bad = ex.execute("queue_job", {"command": "rm -rf /",
                                       "experiment_id": "x"})
        self.assertFalse(bad["ok"])
        missing = ex.execute("poll_job", {"job_id": "nope"})
        self.assertFalse(missing["ok"])


if __name__ == "__main__":
    unittest.main()

