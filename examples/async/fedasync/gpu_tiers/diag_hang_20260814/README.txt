# Hang captured 2026-08-14 01:40
both gpu_xa20_gap10 runs froze at 23:45:03 after 'Sent 42.69 MB payload'
all procs in ep_poll; no errors/exceptions; FS+sockets healthy; failcnt=0 oom_kill=0
max log-gap: serial seed8888 = 8-28s | parallel gap01 (completed) = 145-156s | parallel gap10 = 186-195s then permanent

# Reproduced 2026-08-15 09:30, mix sweep (HANDOFF 4d)
two parallel run_sweep.sh lanes on ONE box (46.5 GB cgroup, 24% headroom, no OOM):
gpu_xa20_gap30_m50_35_15 + gpu_xa20_gap30_m34_33_33 both froze at round ~175,
within seconds of each other; trainers and all 22 client procs in ep_poll at 0%
CPU for 30+ min; failcnt=0, disk 763T free, GPU idle at 2 MiB.
=> parallelism, not memory and not the config: same signature as 2026-08-14.
The mix sweep was restarted SERIALLY (one run at a time) and the two partial run
directories deleted.  hung_mix_*.server.log are the frozen logs.
