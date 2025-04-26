package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_ioError_4_0_Test {

    @Test
    public void testIoError_WithValidException() {
        // Arrange
        Exception mockException = mock(Exception.class);
        when(mockException.getMessage()).thenReturn("I/O error occurred");
        // Act
        String result = AnyWrapperMsgGenerator.ioError(mockException);
        // Assert
        assertEquals("I/O error occurred", result);
    }

    @Test
    public void testIoError_WithNullException() {
        // Arrange
        Exception mockException = null;
        // Act
        String result = AnyWrapperMsgGenerator.ioError(mockException);
        // Assert
        // Assuming error(null) returns "Null exception"
        assertEquals("Null exception", result);
    }

    @Test
    public void testIoError_WithGenericException() {
        // Arrange
        Exception mockException = new Exception("Generic error");
        // Act
        String result = AnyWrapperMsgGenerator.ioError(mockException);
        // Assert
        assertEquals("Generic error", result);
    }
}
