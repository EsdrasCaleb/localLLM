package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Util_StringCompareIgnCase_3_2_Test {

    @Test
    void testStringCompareIgnCase() {
        // Arrange
        String lhs = "Hello";
        String rhs = "world";
        Util util = Mockito.mock(Util.class);
        // Act
        int result = util.StringCompareIgnCase(lhs, rhs);
        // Assert
        Mockito.verify(util).StringCompareIgnCase(lhs, rhs);
        assertEquals(1, result);
    }
}
