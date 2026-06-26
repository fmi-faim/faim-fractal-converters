from collections import defaultdict


def reverse_mapping(
    tcz_to_file: dict[tuple[int, int, int], str],
    *,
    assert_unique_tc: bool = True,
    assert_contiguous_z: bool = True,
) -> dict[str, dict[tuple[int, int], list[int]]]:
    """
    Reverse a (t, c, z) -> filename mapping to filename -> {(t, c): [z, ...]} mapping.

    Parameters
    ----------
    tcz_to_file:
        Input mapping of (t, c, z) tuples to filenames.
    assert_unique_tc:
        If True, raise an error when the same filename maps to more than one (t, c) pair.
    assert_contiguous_z:
        If True, raise an error when the z values for a given (t, c, filename) group
        are not contiguous (i.e. contain gaps).

    Returns
    -------
    Nested mapping: filename -> (t, c) -> sorted list of z values.
    """
    result: dict[str, dict[tuple[int, int], list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for (t, c, z), filename in tcz_to_file.items():
        result[filename][(t, c)].append(z)

    # Sort z lists and run optional checks
    for filename, tc_map in result.items():
        if assert_unique_tc and len(tc_map) > 1:
            raise ValueError(
                f"File {filename!r} maps to multiple (t, c) pairs: {sorted(tc_map)}"
            )

        for tc, z_list in tc_map.items():
            z_list.sort()
            if assert_contiguous_z:
                expected = list(range(z_list[0], z_list[0] + len(z_list)))
                if z_list != expected:
                    raise ValueError(
                        f"File {filename!r}, (t, c)={tc}: z values are not contiguous: {z_list}"
                    )

    return dict(result)  # unwrap outer defaultdict for cleaner repr
