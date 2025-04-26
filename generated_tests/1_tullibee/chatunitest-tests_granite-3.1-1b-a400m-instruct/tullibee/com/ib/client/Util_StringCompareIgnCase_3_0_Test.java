package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Util_StringCompareIgnCase_3_0_Test {

    @Test
    void testStringCompareIgnCase() {
        // Given
        String lhs = "Hello World";
        String rhs = "hello world";
        // When
        int result = Util.StringCompareIgnCase(lhs, rhs);
        // Then
        // Both strings are equal
        assertEquals(0, result);
    }

    @Test
    void testStringCompareIgnCaseWithCaseInsensitive() {
        // Given
        String lhs = "Hello World";
        String rhs = "hello world";
        // When
        int result = Util.StringCompareIgnCase(lhs, rhs);
        // Then
        // Both strings are equal
        assertEquals(0, result);
    }

    @Test
    void testStringCompareIgnCaseWithDifferentCaseInsensitive() {
        // Given
        String lhs = "Hello World";
        String rhs = "Hello World";
        // When
        int result = Util.StringCompareIgnCase(lhs, rhs);
        // Then
        // Both strings are equal
        assertEquals(0, result);
    }
}
