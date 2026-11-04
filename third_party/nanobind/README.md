# nanobind notices

The Python extension is built with nanobind, which `pyproject.toml` pins in
`[build-system] requires`. Its support library is compiled into the extension
statically, and with it the hash table of tsl-robin-map, which nanobind
vendors as a submodule:

```text
nanobind 3.1.0 commit 1852d88cd6632c6ab48ce77be6f4645ad08f200e
robin-map 1.4.0 commit 4ec1bf19c6a96125ea22062f38c2cf5b958e448e
```

`LICENSE` is nanobind's license (BSD-3-Clause) and `robin_map_LICENSE` is
robin-map's (MIT), both from those commits (https://github.com/wjakob/nanobind,
https://github.com/Tessil/robin-map). The wheel metadata lists both as license
files, and the wheel SBOMs list both components
(`scripts/sbom_bundled_components.py`, which reads the lines above). Source
distributions do not include nanobind; building one installs the pinned
version. Updating nanobind means changing the pin, these lines and the two
license files together.
