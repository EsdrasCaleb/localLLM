package com.ib.client;

import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_StringCompareIgnCase_3_1_Test {

    @Test
    public void testStringCompareIgnCase_EqualStrings_ReturnsZero() {
        String lhs = "Hello";
        String rhs = "hello";
        int expected = 0;
        int actual = Util.StringCompareIgnCase(lhs, rhs);
        assertEquals(expected, actual);
    }

    @Test
    public void testStringCompareIgnCase_LowercaseFirstString_ReturnsNegativeValue() {
        String lhs = "hello";
        String rhs = "World";
        int expected = -1;
        int actual = Util.StringCompareIgnCase(lhs, rhs);
        assertEquals(expected, actual);
    }

    @Test
    public void testStringCompareIgnCase_UppercaseFirstString_ReturnsPositiveValue() {
        String lhs = "HELLO";
        String rhs = "world";
        int expected = 1;
        int actual = Util.StringCompareIgnCase(lhs, rhs);
        assertEquals(expected, actual);
    }

    @Test
    public void testStringCompareIgnCase_DifferentLengthStrings_ReturnsCorrectOrder() {
        String lhs = "abc";
        String rhs = "abcd";
        int expected = -1;
        int actual = Util.StringCompareIgnCase(lhs, rhs);
        assertEquals(expected, actual);
    }

    @Test
    public void testStringCompareIgnCase_NullLeftHandedString_ThrowsNullPointerException() {
        String lhs = null;
        String rhs = "hello";
        assertThrows(NullPointerException.class, () -> Util.StringCompareIgnCase(lhs, rhs));
    }

    @Test
    public void testStringCompareIgnCase_NullRightHandedString_ThrowsNullPointerException() {
        String lhs = "hello";
        String rhs = null;
        assertThrows(NullPointerException.class, () -> Util.StringCompareIgnCase(lhs, rhs));
    }

    @Test
    public void testStringCompareIgnCase_EmptyStrings_ReturnsZero() {
        String lhs = "";
        String rhs = "";
        int expected = 0;
        int actual = Util.StringCompareIgnCase(lhs, rhs);
        assertEquals(expected, actual);
    }
}
