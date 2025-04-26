package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class AnyWrapperMsgGenerator_error_0_0_Test {

    @Test
    void testErrorWithException() {
        // Arrange
        Exception mockException = mock(Exception.class);
        when(mockException.getMessage()).thenReturn("Mocked exception message");
        // Act
        String result = AnyWrapperMsgGenerator.error(mockException);
        // Assert
        assertEquals("Error - Mocked exception message", result);
    }
}
