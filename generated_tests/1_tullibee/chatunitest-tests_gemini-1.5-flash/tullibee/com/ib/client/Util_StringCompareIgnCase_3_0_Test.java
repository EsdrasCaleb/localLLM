package com.ib.client;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_StringCompareIgnCase_3_0_Test {

    @Test
    void testStringCompareIgnCase_NullStrings() {
        assertEquals(0, Util.StringCompareIgnCase(null, null));
        assertEquals(-1, Util.StringCompareIgnCase(null, "test"));
        assertEquals(1, Util.StringCompareIgnCase("test", null));
    }

    @Test
    void testStringCompareIgnCase_EmptyStrings() {
        assertEquals(0, Util.StringCompareIgnCase("", ""));
        assertEquals(-1, Util.StringCompareIgnCase("", "test"));
        assertEquals(1, Util.StringCompareIgnCase("test", ""));
    }

    @Test
    void testStringCompareIgnCase_EqualStrings() {
        assertEquals(0, Util.StringCompareIgnCase("test", "test"));
        assertEquals(0, Util.StringCompareIgnCase("Test", "test"));
        assertEquals(0, Util.StringCompareIgnCase("TEST", "Test"));
    }

    @Test
    void testStringCompareIgnCase_DifferentStrings() {
        assertEquals(-1, Util.StringCompareIgnCase("apple", "banana"));
        assertEquals(1, Util.StringCompareIgnCase("banana", "apple"));
        assertEquals(-1, Util.StringCompareIgnCase("apple", "Apple"));
        assertEquals(1, Util.StringCompareIgnCase("Apple", "apple"));
    }

    @Test
    void testStringCompareIgnCase_WhitespaceStrings() {
        assertEquals(0, Util.StringCompareIgnCase("  test  ", "  Test  "));
        assertEquals(0, Util.StringCompareIgnCase("test", "  test  "));
        assertEquals(0, Util.StringCompareIgnCase("  test  ", "test"));
    }

    @Test
    void testNormalizeString_Reflection() throws Exception {
        Method normalizeStringMethod = Util.class.getDeclaredMethod("NormalizeString", String.class);
        normalizeStringMethod.setAccessible(true);
        assertEquals("test", normalizeStringMethod.invoke(null, "  test  "));
        assertEquals("test", normalizeStringMethod.invoke(null, "TEST"));
        assertEquals("", normalizeStringMethod.invoke(null, ""));
        assertEquals(null, normalizeStringMethod.invoke(null, null));
    }
}
