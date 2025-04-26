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
    public void testStringCompare_EqualStrings_ReturnZero() {
        String lhs = "apple";
        String rhs = "apple";
        assertEquals(0, Util.StringCompare(lhs, rhs));
    }

    @Test
    public void testStringCompare_String1GreaterThanString2_ReturnPositiveValue() {
        String lhs = "banana";
        String rhs = "apple";
        assertEquals(1, Util.StringCompare(lhs, rhs));
    }

    @Test
    public void testStringCompare_String2GreaterThanString1_ReturnNegativeValue() {
        String lhs = "apple";
        String rhs = "banana";
        assertEquals(-1, Util.StringCompare(lhs, rhs));
    }

    @Test
    public void testStringCompare_NullString1_ThrowsNullPointerException() {
        String lhs = null;
        String rhs = "apple";
        assertThrows(NullPointerException.class, () -> Util.StringCompare(lhs, rhs));
    }

    @Test
    public void testStringCompare_NullString2_ThrowsNullPointerException() {
        String lhs = "apple";
        String rhs = null;
        assertThrows(NullPointerException.class, () -> Util.StringCompare(lhs, rhs));
    }
}
