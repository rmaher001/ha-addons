"""The add-on is a static thing: one pinned image, one config file, no
options. These tests pin the shape so a change to it is deliberate."""
import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
ADDON = ROOT / "alloy"
STORE = "http://192.168.20.222:9428/insert/loki/api/v1/push"


def read(name):
    return (ADDON / name).read_text()


def test_the_repository_names_itself():
    repo = yaml.safe_load((ROOT / "repository.yaml").read_text())
    assert repo["url"] == "https://github.com/rmaher001/ha-addons"
    assert repo["name"] and repo["maintainer"]


def test_the_addon_reads_the_host_journal_and_nothing_else():
    cfg = yaml.safe_load(read("config.yaml"))
    assert cfg["slug"] == "alloy" and cfg["arch"] == ["amd64"]
    assert cfg["journald"] is True, "the host journal is the whole point"
    assert "map" not in cfg, "no HA directory is mounted"
    assert "options" not in cfg and "schema" not in cfg, "config is born in git, never typed in the UI"
    assert "host_network" not in cfg and "privileged" not in cfg and cfg.get("apparmor", True) is not False
    assert cfg["init"] is True, "the image has no init of its own"
    assert cfg["startup"] == "services" and cfg["boot"] == "auto"


def test_the_debug_port_is_not_exposed_but_the_watchdog_still_reaches_it():
    cfg = yaml.safe_load(read("config.yaml"))
    assert cfg["ports"] == {"12345/tcp": None}
    assert cfg["watchdog"] == "http://[HOST]:[PORT:12345]/-/ready"


def alloy_command():
    """The one line that starts Alloy: run.sh's exec."""
    return re.search(r"^exec /bin/alloy run (.*)$", read("run.sh"), re.M).group(1)


def test_the_image_is_pinned_to_one_alloy_release_and_started_by_our_script():
    df = read("Dockerfile")
    froms = re.findall(r"^FROM (\S+)", df, re.M)
    assert len(froms) == 1 and re.fullmatch(r"grafana/alloy:v\d+\.\d+\.\d+", froms[0]), froms
    assert "COPY config.alloy /etc/alloy/config.alloy" in df
    assert 'ENTRYPOINT ["/run.sh"]' in df and "CMD" not in df
    cmd = alloy_command()
    for arg in ("--disable-reporting", "--storage.path=/data/alloy", "/etc/alloy/config.alloy"):
        assert arg in cmd, arg


def test_the_journal_path_is_chosen_at_start_never_left_empty():
    """An empty `path` opens the journal 'local only' - the container's own
    machine-id - and reads nothing (found live 2026-09-05: zero lines).
    run.sh picks whichever of the two host directories holds the journal."""
    sh = read("run.sh")
    assert sh.startswith("#!/bin/sh"), "the Alloy image has sh, nothing more"
    assert "/var/log/journal" in sh and "/run/log/journal" in sh
    assert "export JOURNAL_PATH" in sh
    assert 'path          = sys.env("JOURNAL_PATH")' in read("config.alloy")


def test_the_version_is_the_changelogs_newest_entry():
    cfg = yaml.safe_load(read("config.yaml"))
    top = re.search(r"^## (\S+)", read("CHANGELOG.md"), re.M).group(1)
    assert cfg["version"] == top


def test_the_config_ships_the_journal_to_the_estates_store_with_its_labels():
    c = read("config.alloy")
    assert 'loki.source.journal' in c and 'max_age' in c
    for label in ('job        = "journal"', 'host       = "homeassistant"', 'vmid       = "200"',
                  'node       = "proxmox2"', 'guest_type = "vm"'):
        assert label in c, label
    assert f'url = "{STORE}"' in c
    assert "wal {" in c and "enabled         = true" in c
    assert 'source   = "received_at"' in c, "event time is the entry time; ingestion time rides along"


def test_container_lines_carry_stream_and_host_units_carry_severity():
    c = read("config.alloy")
    assert 'target_label  = "container"' in c
    assert 'target_label  = "stream"' in c
    assert 'regex         = "@(.*)"' in c and 'target_label  = "severity"' in c
    assert 'target_label  = "level"' not in c


def test_transient_units_are_dropped_before_they_become_a_label():
    c = read("config.alloy")
    assert "session-[a-z]*[0-9]+\\\\.scope" in c and 'action        = "drop"' in c


def test_an_experimental_feature_in_the_config_has_its_flag_on_the_command_line():
    """loki.write's wal block is EXPERIMENTAL in Alloy 1.19: without
    --stability.level=experimental Alloy refuses the config and the add-on
    crash-loops, shipping nothing (review, 2026-09-05)."""
    if "wal {" in read("config.alloy"):
        assert "--stability.level=experimental" in alloy_command()


def test_the_docs_query_the_severity_values_alloy_really_writes():
    # journald priority 3 is spelled "error" by Alloy, never "err"
    docs = read("DOCS.md")
    assert "severity:(error OR crit)" in docs and "severity:(err " not in docs
    assert '(err|error)' not in read("config.alloy")
