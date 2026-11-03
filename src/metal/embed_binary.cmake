# Writes INPUT's bytes to OUTPUT as a comma-separated list of hex literals for
# inclusion in a C++ array initializer. Usage:
#   cmake -DINPUT=<file> -DOUTPUT=<file> -P embed_binary.cmake
file(READ "${INPUT}" WOLFGANG_EMBED_HEX HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," WOLFGANG_EMBED_BYTES "${WOLFGANG_EMBED_HEX}")
file(WRITE "${OUTPUT}" "${WOLFGANG_EMBED_BYTES}\n")
