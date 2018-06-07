print("<table>")

for i in range(1, 1099):
    print("<tr><td> </td>\
    <td>source</td>\
    <td>gt</td>\
    <td>Original</td>\
    <td>Res6</td>\
    <td>Res9</td>\
    <td>UNet</td>\
    <td>Dual-D</td>\
    <td>DeepDual-D</td>\
    <td>WGAN</td>\
    </tr>")

    print("<tr>\
    <td>{}</td>\
    <td><img src='images/real_A/{}_A.png' /></td> \
    <td><img width=256 height=256 src='images/gt_B/{}_B.jpg' />\
    <td><img src='images/orig_B/{}_A.png' /></td> \
    <td><img src='images/res6_B/B_gen_test_{}.png' /> \
    <td><img src='images/res9_B/B_gen_test_{}.png' /> \
    <td><img src='images/unet/B_gen_test_{}.png' /> \
    <td><img src='images/dual/B_gen_test_{}.png' /> \
    <td><img src='images/dualdeep/B_gen_test_{}.png' /> \
    <td><img src='images/wgan/B_gen_test_{}.png' /> \
    </tr>".format(i, i, i, i, i, i, i,i,i,i))

print("</table>")