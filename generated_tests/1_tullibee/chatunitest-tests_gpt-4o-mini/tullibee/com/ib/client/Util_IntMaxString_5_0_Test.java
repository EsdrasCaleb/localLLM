package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_IntMaxString_5_0_Test {

    @Test
    public void testIntMaxString_WhenValueIsIntegerMaxValue_ShouldReturnEmptyString() {
        // Arrange
        int input = Integer.MAX_VALUE;
        // Act
        String result = Util.IntMaxString(input);
        // Assert
        assertEquals("", result);
    }

    @Test
    public void testIntMaxString_WhenValueIsLessThanIntegerMaxValue_ShouldReturnStringRepresentation() {
        // Arrange
        int input = 123;
        // Act
        String result = Util.IntMaxString(input);
        // Assert
        assertEquals("123", result);
    }

    @Test
    public void testIntMaxString_WhenValueIsNegative_ShouldReturnStringRepresentation() {
        // Arrange
        int input = -456;
        // Act
        String result = Util.IntMaxString(input);
        // Assert
        assertEquals("-456", result);
    }
}
