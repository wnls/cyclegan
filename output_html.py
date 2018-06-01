print("<table>")
print("<tr><td> </td>\
        <td>source</td>\
        <td>original</td>\
        <td>res6</td>\
        <td>res9</td>\
        <td>gt</td>\
        </tr>")
for i in range(1, 1099):
    print("<tr><td>{}</td><td><img src='images/real_A/{}_A.png' /></td> \
          <td><img src='images/orig_B/{}_A.png' /></td> \
          <td><img src='images/res6_B/B_gen_test_{}.png' /> \
          <td><img src='images/res9_B/B_gen_test_{}.png' /> \
          <td><img width=256 height=256 src='images/gt_B/{}_B.jpg' /></tr>".format(i, i, i, i, i, i))
print("</table>")