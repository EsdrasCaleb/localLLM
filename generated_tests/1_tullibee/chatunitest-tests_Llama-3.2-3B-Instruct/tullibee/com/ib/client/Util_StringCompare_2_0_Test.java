package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_StringCompare_2_0_Test {

    @Test
    public void testStringCompare_EqualStrings_ReturnsZero() {
        String lhs = "Hello";
        String rhs = "Hello";
        int expected = 0;
        int actual = Util.StringCompare(lhs, rhs);
        assertEquals(expected, actual);
    }

    @Test
    public void testStringCompare_DifferentStrings_ReturnsNegativeIndex() {
        String lhs = "Hello";
        String rhs = "World";
        int expected = -1;
        int actual = Util.StringCompare(lhs, rhs);
        assertEquals(expected, actual);
    }

    @Test
    public void testStringCompare_NullLeftHandSide_ThrowsNullPointerException() {
        String lhs = null;
        String rhs = "World";
        assertThrows(NullPointerException.class, () -> Util.StringCompare(lhs, rhs));
    }

    @Test
    public void testStringCompare_NullRightHandSide_ThrowsNullPointerException() {
        String lhs = "Hello";
        String rhs = null;
        assertThrows(NullPointerException.class, () -> Util.StringCompare(lhs, rhs));
    }

    @Test
    public void testStringCompare_EmptyString_ReturnsNegativeIndex() {
        String lhs = "";
        String rhs = "World";
        int expected = -1;
        int actual = Util.StringCompare(lhs, rhs);
        assertEquals(expected, actual);
    }

    @Test
    public void testStringCompare_EmptyStringLeft_HandlesItCorrectly() {
        String lhs = "";
        String rhs = "World";
        int expected = -1;
        int actual = Util.StringCompare(lhs, rhs);
        assertEquals(expected, actual);
    }

    @Test
    public void testStringCompare_EmptyStringRight_HandlesItCorrectly() {
        String lhs = "Hello";
        String rhs = "";
        int expected = 1;
        int actual = Util.StringCompare(lhs, rhs);
        assertEquals(expected, actual);
    }
}
