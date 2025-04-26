package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_NormalizeString_1_0_Test {

    @Test
    public void testNormalizeStringWithNull() {
        // Arrange
        String input = null;
        String expectedOutput = "";
        // Act
        String actualOutput = Util.NormalizeString(input);
        // Assert
        assertEquals(expectedOutput, actualOutput, "The method should return an empty string when input is null");
    }

    @Test
    public void testNormalizeStringWithEmptyString() {
        // Arrange
        String input = "";
        String expectedOutput = "";
        // Act
        String actualOutput = Util.NormalizeString(input);
        // Assert
        assertEquals(expectedOutput, actualOutput, "The method should return an empty string when input is an empty string");
    }

    @Test
    public void testNormalizeStringWithNonEmptyString() {
        // Arrange
        String input = "testString";
        String expectedOutput = "testString";
        // Act
        String actualOutput = Util.NormalizeString(input);
        // Assert
        assertEquals(expectedOutput, actualOutput, "The method should return the original string when input is a non-empty string");
    }
}
