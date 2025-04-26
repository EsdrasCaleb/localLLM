package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Util_DoubleMaxString_6_1_Test {

    @Test
    public void testDoubleMaxString() {
        // Arrange
        double value = Double.MAX_VALUE;
        String expectedResult = "";
        // Act
        String result = Util.DoubleMaxString(value);
        // Assert
        assertEquals(expectedResult, result);
    }
}
