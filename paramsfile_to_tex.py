#!-*-coding:utf-8-*-
#パラメータをtexに書きなおすのがめんどくさすぎたのでスクリプトを書く

import numpy as np


def numpy_matrix_to_tex_pmatrix(np_mat):
    pmatrix = "$\\begin{pmatrix}"
    for row_i in range(np_mat.shape[0]):
        for column_i in range(np_mat.shape[1]):
            pmatrix += "%s"%float("%1g"%np_mat[row_i,column_i])
            pmatrix += " & "
        pmatrix = pmatrix.rstrip("& ")
        pmatrix += " \\\\ "
    pmatrix = pmatrix.rstrip("\\ ")
    pmatrix += "\\end{pmatrix}$"
    return pmatrix


if __name__ == '__main__':
    import sys
    sys.path.append('params/')

    params_module_name = sys.argv[1]

    methods = ["mu1","mu2","Wx1","Wx2", "Wt1","Wt2", "Psi1", "Psi2"]
#    methods = ["mu1","mu2","Wx1","Wx2"]
#    methods = ["Wt1","Wt2", "Psi1", "Psi2"]
    module = __import__(params_module_name, globals(), locals(), methods, -1)
    for x in methods:
        locals()[x]=getattr(module,x)
#    params = __import__(params_module_name)
#    numpy_matrix_to_tex_pmatrix(params.mu1)

    table_row = " & "

#    for temp_param in (mu1,mu2,Wx1,Wx2, Wt1,Wt2, Psi1, Psi2):
#    for temp_param in (mu1,mu2,Wx1,Wx2):
    for temp_param in (Wt1,Wt2, Psi1, Psi2):
        table_row += numpy_matrix_to_tex_pmatrix(temp_param)
        table_row += " & "
    print table_row
