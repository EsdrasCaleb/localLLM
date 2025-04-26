package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class Util_StringCompareIgnCase_3_1_Test {

    @InjectMocks
    private Util util;

    @Test
    public void testStringCompareIgnCase() {
        // Test case 1: Both strings are null
        assertEquals(0, Util.StringCompareIgnCase(null, null));
        // Test case 2: First string is null
        assertTrue(Util.StringCompareIgnCase(null, "test") < 0);
        // Test case 3: Second string is null
        assertTrue(Util.StringCompareIgnCase("test", null) > 0);
        // Test case 4: Both strings are empty
        assertEquals(0, Util.StringCompareIgnCase("", ""));
        // Test case 5: First string is empty
        assertTrue(Util.StringCompareIgnCase("", "test") < 0);
        // Test case 6: Second string is empty
        assertTrue(Util.StringCompareIgnCase("test", "") > 0);
        // Test case 7: Both strings are equal
        assertEquals(0, Util.StringCompareIgnCase("test", "test"));
        // Test case 8: Both strings are equal (case insensitive)
        assertEquals(0, Util.StringCompareIgnCase("Test", "test"));
        // Test case 9: First string is lexicographically less than second string
        assertTrue(Util.StringCompareIgnCase("apple", "banana") < 0);
        // Test case 10: First string is lexicographically greater than second string
        assertTrue(Util.StringCompareIgnCase("banana", "apple") > 0);
        // Test case 11: Both strings are equal (with normalization)
        assertEquals(0, Util.StringCompareIgnCase("Test123", "test123"));
        // Test case 12: First string is lexicographically less than second string (with normalization)
        assertTrue(Util.StringCompareIgnCase("Apple123", "Banana456") < 0);
        // Test case 13: First string is lexicographically greater than second string (with normalization)
        assertTrue(Util.StringCompareIgnCase("Banana456", "Apple123") > 0);
    }
}
