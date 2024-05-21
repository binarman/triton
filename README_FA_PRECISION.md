### build mfma example

`hipcc  mfma.cpp --offload-arch=gfx942 -Xclang -fdenormal-fp-math-f32=preserve-sign,preserve-sign -fdenormal-fp-math=preserve-sign,preserve-sign`
