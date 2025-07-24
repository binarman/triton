### In elements

if (elemBitWidth == 16 && kWidth == 4) {
    if (write_width < 8) {    // 1/2 banks per access
      pad1 = k:+4
      pad2_interval = k * 16
      pad2_padding = max(512 / nonk, 4)
    } else {                  // 4 banks per access
      pad1 = k:+8
      pad2_interval = k * 8
      pad2_padding = max(512 / nonk, 8)
    }
}

if (elemBitWidth == 8 && kWidth == 8) {
    pad1 = k:+8
    pad2_interval = k * 16
    pad2_padding = max(min(1028 / nonk, 16), 8)
}

### In bytes

write_elems -> 2, 4, 8
read_elems -> kWidth -> 4, 8

elem_width = elemBitWidth / 8
write_width = elem_width * write_elems
read_width = kWidth * elem_width

if (elem_width == 2 && read_width == 8) {
    if (write_width < 16) {
      pad1 = k*2:+8
      pad2_interval = k * 32
      pad2_padding = max(1024 / nonk, 8)
    } else {
      pad1 = k*2:+16
      pad2_interval = k * 16
      pad2_padding = max(1024 / nonk, 16)
    }
}

if (elem_width == 1 && read_width == 8) {
    pad1 = k:+8
    pad2_interval = k * 16
    pad2_padding = max(min(1028 / nonk, 16), 8)
}

->

if (write_width < 16) {
  pad1 = k*elem_width:+8
  pad2_interval = k * 16 * elem_width
  pad2_padding = max(min(1028 / nonk, 16), 8)
} else {
  pad1 = k*elem_width:+16
  pad2_interval = k * 16 * elem_width
  pad2_padding = max(1024 / nonk, 16)
}
