package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_DoubleMaxString_6_0_Test {

    @Test
    public void testDoubleMaxString_withDoubleMaxValue() {
        // Arrange
        double input = Double.MAX_VALUE;
        String expectedOutput = "";
        // Act
        String actualOutput = Util.DoubleMaxString(input);
        // Assert
        assertEquals(expectedOutput, actualOutput, "The method should return an empty string when the input is Double.MAX_VALUE");
    }

    @Test
    public void testDoubleMaxString_withPositiveDoubleValue() {
        // Arrange
        double input = 123.456;
        String expectedOutput = "123.456";
        // Act
        String actualOutput = Util.DoubleMaxString(input);
        // Assert
        assertEquals(expectedOutput, actualOutput, "The method should return the string representation of the input when it is not Double.MAX_VALUE");
    }

    @Test
    public void testDoubleMaxString_withNegativeDoubleValue() {
        // Arrange
        double input = -123.456;
        String expectedOutput = "-123.456";
        // Act
        String actualOutput = Util.DoubleMaxString(input);
        // Assert
        assertEquals(expectedOutput, actualOutput, "The method should return the string representation of the input when it is not Double.MAX_VALUE");
    }

    @Test
    public void testDoubleMaxString_withZero() {
        // Arrange
        double input = 0.0;
        String expectedOutput = "0.0";
        // Act
        String actualOutput = Util.DoubleMaxString(input);
        // Assert
        assertEquals(expectedOutput, actualOutput, "The method should return the string representation of the input when it is not Double.MAX_VALUE");
    }

    @Test
    public void testDoubleMaxString_withPositiveInfinity() {
        // Arrange
        double input = Double.POSITIVE_INFINITY;
        String expectedOutput = "" + Double.POSITIVE_INFINITY;
        // Act
        String actualOutput = Util.DoubleMaxString(input);
        // Assert
        assertEquals(expectedOutput, actualOutput, "The method should return the string representation of the input when it is not Double.MAX_VALUE");
    }

    @Test
    public void testDoubleMaxString_withNegativeInfinity() {
        // Arrange
        double input = Double.NEGATIVE_INFINITY;
        String expectedOutput = "" + Double.NEGATIVE_INFINITY;
        // Act
        String actualOutput = Util.DoubleMaxString(input);
        // Assert
        assertEquals(expectedOutput, actualOutput, "The method should return the string representation of the input when it is not Double.MAX_VALUE");
    }
}
