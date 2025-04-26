package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_StringCompareIgnCase_3_0_Test {

    @Test
    public void testStringCompareIgnCase_EqualStrings() {
        assertEquals(0, Util.StringCompareIgnCase("test", "TEST"));
    }

    @Test
    public void testStringCompareIgnCase_LhsLessThanRhs() {
        assertEquals(-1, Util.StringCompareIgnCase("apple", "Banana"));
    }

    @Test
    public void testStringCompareIgnCase_LhsGreaterThanRhs() {
        assertEquals(1, Util.StringCompareIgnCase("Banana", "apple"));
    }

    @Test
    public void testStringCompareIgnCase_EmptyStrings() {
        assertEquals(0, Util.StringCompareIgnCase("", ""));
    }

    @Test
    public void testStringCompareIgnCase_LhsEmptyRhsNotEmpty() {
        assertEquals(-1, Util.StringCompareIgnCase("", "test"));
    }

    @Test
    public void testStringCompareIgnCase_LhsNotEmptyRhsEmpty() {
        assertEquals(1, Util.StringCompareIgnCase("test", ""));
    }

    @Test
    public void testStringCompareIgnCase_NullStrings() {
        assertEquals(0, Util.StringCompareIgnCase(null, null));
    }

    @Test
    public void testStringCompareIgnCase_NullLhs() {
        assertEquals(-1, Util.StringCompareIgnCase(null, "test"));
    }

    @Test
    public void testStringCompareIgnCase_NullRhs() {
        assertEquals(1, Util.StringCompareIgnCase("test", null));
    }
}
