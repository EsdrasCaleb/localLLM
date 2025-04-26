package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Util_NormalizeString_1_0_Test {

    private Util util;

    @BeforeEach
    public void setUp() {
        this.util = new Util();
    }

    @Test
    public void testNormalizeStringWithNullInput() {
        // Arrange
        String input = null;
        // Act
        String result = util.NormalizeString(input);
        // Assert
        assertEquals("", result);
    }

    @Test
    public void testNormalizeStringWithNonEmptyInput() {
        // Arrange
        String input = "   Hello World!   ";
        // Act
        String result = util.NormalizeString(input);
        // Assert
        assertEquals("Hello World!", result);
    }

    @Test
    public void testNormalizeStringWithNoWhitespace() {
        // Arrange
        String input = "Java";
        // Act
        String result = util.NormalizeString(input);
        // Assert
        assertEquals("Java", result);
    }
}
