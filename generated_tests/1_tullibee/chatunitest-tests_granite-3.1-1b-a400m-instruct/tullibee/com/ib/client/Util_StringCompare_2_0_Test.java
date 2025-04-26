package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Util_StringCompare_2_0_Test {

    @Test
    void testStringCompare() {
        Util util = new Util();
        // Test case 1: Strings are equal
        assertEquals(0, util.StringCompare("hello", "hello"));
        // Test case 2: Strings are not equal
        assertEquals(-1, util.StringCompare("hello", "world"));
        // Test case 3: Strings are compared lexicographically
        assertEquals(0, util.StringCompare("hello", "world"));
        // Test case 4: Strings are compared lexicographically and are equal
        assertEquals(0, util.StringCompare("hello", "hello"));
        // Test case 5: Strings are compared lexicographically and are not equal
        assertEquals(-1, util.StringCompare("hello", "world"));
    }
}
