from PIL import Image
import struct
import ctypes

input_filename = input("input filename: ")
output_filename = input("output filename: ")

img = Image.open(input_filename)
(w, h) = img.size[0:2]
pix = img.load()
buff = ctypes.create_string_buffer(4 * w * h)
offset = 0
for j in range(h):
	for i in range(w):
		r = bytes((pix[i, j][0],))
		g = bytes((pix[i, j][1],))
		b = bytes((pix[i, j][2],))
		a = bytes((pix[i, j][3],))
		struct.pack_into('cccc', buff, offset, r, g, b, a)
		offset += 4
out = open(output_filename, 'wb')
out.write(struct.pack('ii', w, h))
out.write(buff.raw)
out.close()
