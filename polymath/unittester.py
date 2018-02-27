################################################################################
# polymath/unittester.py
#
# Mark R. Showalter, PDS Rings Node, SETI Institute
################################################################################
# Execute unit tests from command line...
################################################################################

if __name__ == '__main__':

    from unit_tests.test_boolean                import *

    from unit_tests.test_empty                  import *

    from unit_tests.test_indices                import *
    
    from unit_tests.test_matrix_column_vectors  import *
    from unit_tests.test_matrix_inverse         import *
    from unit_tests.test_matrix_is_diagonal     import *
    from unit_tests.test_matrix_misc            import *
    from unit_tests.test_matrix_ops             import *
    from unit_tests.test_matrix_row_vectors     import *
    from unit_tests.test_matrix_unitary         import *

    from unit_tests.test_matrix3_quaternion     import *
    from unit_tests.test_matrix3_euler          import *
    from unit_tests.test_matrix3_twovec         import *

    from unit_tests.test_pair_as_pair           import *
    from unit_tests.test_pair_clip2d            import *
    from unit_tests.test_pair_misc              import *
    from unit_tests.test_pair_swapxy            import *

    from unit_tests.test_quaternion             import *
    from unit_tests.test_quaternion_euler       import *
    from unit_tests.test_quaternion_matrix3     import *
    from unit_tests.test_quaternion_parts       import *
    from unit_tests.test_quaternion_ops         import *

    from unit_tests.test_qube_all               import *
    from unit_tests.test_qube_any               import *
    from unit_tests.test_qube_derivs            import *
    from unit_tests.test_qube_getitem           import *
    from unit_tests.test_qube_identity          import *
    from unit_tests.test_qube_items             import *
    from unit_tests.test_qube_types             import *
    from unit_tests.test_qube_masking           import *
    from unit_tests.test_qube_readonly          import *
    from unit_tests.test_qube_reshaping         import *
    from unit_tests.test_qube_setitem           import *
    from unit_tests.test_qube_shrink            import *
    from unit_tests.test_qube_stack             import *
    from unit_tests.test_qube_units             import *
    from unit_tests.test_qube_zero              import *

    from unit_tests.test_scalar_arccos          import *
    from unit_tests.test_scalar_arcsin          import *
    from unit_tests.test_scalar_arctan          import *
    from unit_tests.test_scalar_arctan2         import *
    from unit_tests.test_scalar_as_index        import *
    from unit_tests.test_scalar_as_scalar       import *
    from unit_tests.test_scalar_cos             import *
    from unit_tests.test_scalar_exp             import *
    from unit_tests.test_scalar_frac            import *
    from unit_tests.test_scalar_int             import *
    from unit_tests.test_scalar_log             import *
    from unit_tests.test_scalar_max             import *
    from unit_tests.test_scalar_maximum         import *
    from unit_tests.test_scalar_mean            import *
    from unit_tests.test_scalar_median          import *
    from unit_tests.test_scalar_min             import *
    from unit_tests.test_scalar_minimum         import *
    from unit_tests.test_scalar_misc            import *
    from unit_tests.test_scalar_ops             import *
    from unit_tests.test_scalar_reciprocal      import *
    from unit_tests.test_scalar_sign            import *
    from unit_tests.test_scalar_sin             import *
    from unit_tests.test_scalar_sum             import *
    from unit_tests.test_scalar_sqrt            import *
    from unit_tests.test_scalar_tan             import *

    from unit_tests.test_vector_as_column       import *
    from unit_tests.test_vector_as_diagonal     import *
    from unit_tests.test_vector_as_index        import *
    from unit_tests.test_vector_as_row          import *
    from unit_tests.test_vector_as_vector       import *
    from unit_tests.test_vector_cross_2x2       import *
    from unit_tests.test_vector_cross_3x3       import *
    from unit_tests.test_vector_dot             import *
    from unit_tests.test_vector_element_div     import *
    from unit_tests.test_vector_element_mul     import *
    from unit_tests.test_vector_masking         import *
    from unit_tests.test_vector_norm            import *
    from unit_tests.test_vector_norm_sq         import *
    from unit_tests.test_vector_ops             import *
    from unit_tests.test_vector_outer           import *
    from unit_tests.test_vector_perp            import *
    from unit_tests.test_vector_proj            import *
    from unit_tests.test_vector_scalars         import *
    from unit_tests.test_vector_sep             import *
    from unit_tests.test_vector_ucross          import *
    from unit_tests.test_vector_unit            import *

    from unit_tests.test_vector3                import *
    from unit_tests.test_vector3_misc           import *

    from unit_tests.test_units                  import *

    unittest.main(verbosity=2)

################################################################################
