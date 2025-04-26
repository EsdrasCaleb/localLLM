package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_error_1_1_Test {

    @Test
    public void testError() {
        // Arrange
        String expected = "error";
        String testString = "test";
        // Act
        String result = AnyWrapperMsgGenerator.error(testString);
        // Assert
        assertEquals(expected, result);
    }
}
