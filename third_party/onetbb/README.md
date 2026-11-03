# oneTBB notices

Release CPU wheels bundle the oneTBB runtime library, built by
`scripts/build_onetbb.sh` from oneTBB `v2022.3.0` (commit
`f1862f38f83568d96e814e469ab61f88336cc595`). These are that commit's
`LICENSE.txt` (Apache-2.0) and `third-party-programs.txt` (which includes the
BSD-3-Clause ITT Notify notice). The build fails if the cloned copies differ
from these files, and the wheel metadata lists them as license files. Source
distributions and source builds do not include oneTBB.
