# New unified Bumi x86_64 vendor libraries

Place the x86_64 build of the **new unified Bumi hw** here.

Required:

- `libhw_controller.a`

Optional, if supplied together with the new x86_64 hw package:

- `libsoem.a`
- `libyaml-cppd.a`

The Bumi headers are shared by x86_64 and aarch64 and live in `../../include/`.
Do not copy the legacy Bumi hw library into this directory because its ABI is different.
