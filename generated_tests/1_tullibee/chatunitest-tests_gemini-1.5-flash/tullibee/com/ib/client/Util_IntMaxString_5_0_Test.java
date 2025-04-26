package com.ib.client;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Util_IntMaxString_5_0_Test {

    @Test
    void testIntMaxString() {
        assertEquals("", Util.IntMaxString(Integer.MAX_VALUE));
        assertEquals("123", Util.IntMaxString(123));
        assertEquals("-42", Util.IntMaxString(-42));
        assertEquals("0", Util.IntMaxString(0));
        assertEquals("2147483646", Util.IntMaxString(Integer.MAX_VALUE - 1));
    }

    @Test
    void testStringIsEmpty() {
        assertTrue(Util.StringIsEmpty(null));
        assertTrue(Util.StringIsEmpty(""));
        assertFalse(Util.StringIsEmpty("abc"));
        assertFalse(Util.StringIsEmpty(" "));
    }

    @Test
    void testNormalizeString() {
        assertEquals("", Util.NormalizeString(null));
        assertEquals("abc", Util.NormalizeString("abc"));
        assertEquals(" ", Util.NormalizeString(" "));
    }

    @Test
    void testStringCompare() {
        assertEquals(0, Util.StringCompare("abc", "abc"));
        assertEquals(1, Util.StringCompare("abc", "abd"));
        assertEquals(-1, Util.StringCompare("abc", "abb"));
        assertEquals(0, Util.StringCompare(null, null));
        assertEquals(1, Util.StringCompare("abc", null));
        assertEquals(-1, Util.StringCompare(null, "abc"));
    }

    @Test
    void testStringCompareIgnoreCase() {
        assertEquals(0, Util.StringCompareIgnCase("abc", "ABC"));
        assertEquals(1, Util.StringCompareIgnCase("abc", "abd"));
        assertEquals(-1, Util.StringCompareIgnCase("abc", "abb"));
        assertEquals(0, Util.StringCompareIgnCase(null, null));
        assertEquals(1, Util.StringCompareIgnCase("abc", null));
        assertEquals(-1, Util.StringCompareIgnCase(null, "abc"));
    }

    @Test
    void testVectorEqualsUnordered() {
        Vector<Integer> v1 = new Vector<>();
        v1.add(1);
        v1.add(2);
        v1.add(3);
        Vector<Integer> v2 = new Vector<>();
        v2.add(3);
        v2.add(1);
        v2.add(2);
        Vector<Integer> v3 = new Vector<>();
        v3.add(1);
        v3.add(2);
        v3.add(4);
        assertTrue(Util.VectorEqualsUnordered(v1, v2));
        assertFalse(Util.VectorEqualsUnordered(v1, v3));
        assertTrue(Util.VectorEqualsUnordered(null, null));
        assertFalse(Util.VectorEqualsUnordered(v1, null));
        assertFalse(Util.VectorEqualsUnordered(null, v1));
        assertTrue(Util.VectorEqualsUnordered(new Vector<>(), new Vector<>()));
    }

    @Test
    void testDoubleMaxString() {
        assertEquals("", Util.DoubleMaxString(Double.MAX_VALUE));
        assertEquals("123.45", Util.DoubleMaxString(123.45));
        assertEquals("-42.0", Util.DoubleMaxString(-42.0));
        assertEquals("0.0", Util.DoubleMaxString(0.0));
        assertEquals("2.147483647E9", Util.DoubleMaxString(2147483647.0));
    }
}
