package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class Util_NormalizeString_1_3_Test {

    @Mock
    private Util util;

    @InjectMocks
    private Util testUtil;

    @Test
    public void testNormalizeString() {
        // Arrange
        String input = "Hello, World!";
        // Act
        String expectedOutput = "Hello, World!";
        String actualOutput = testUtil.NormalizeString(input);
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }
}
