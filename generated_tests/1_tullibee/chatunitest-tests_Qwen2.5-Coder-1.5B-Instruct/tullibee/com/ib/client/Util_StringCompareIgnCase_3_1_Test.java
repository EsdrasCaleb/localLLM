package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Util_StringCompareIgnCase_3_1_Test {

    @Test
    void shouldReturnZeroWhenBothStringsAreEqual() {
        // Arrange
        String lhs = "Hello";
        String rhs = "hello";
        // Act
        int result = Util.StringCompareIgnCase(lhs, rhs);
        // Assert
        assertEquals(0, result);
    }

    @Test
    void shouldReturnNegativeValueWhenLeftStringIsLessThanRightString() {
        // Arrange
        String lhs = "apple";
        String rhs = "banana";
        // Act
        int result = Util.StringCompareIgnCase(lhs, rhs);
        // Assert
        assertEquals(-1, result);
    }

    @Test
    void shouldReturnPositiveValueWhenLeftStringIsGreaterThanRightString() {
        // Arrange
        String lhs = "cherry";
        String rhs = "date";
        // Act
        int result = Util.StringCompareIgnCase(lhs, rhs);
        // Assert
        assertEquals(1, result);
    }
}
