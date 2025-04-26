package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class AnyWrapperMsgGenerator_ioError_4_0_Test {

    @Test
    public void testIoError() throws Exception {
        // Arrange
        AnyWrapperMsgGenerator obj = new AnyWrapperMsgGenerator();
        Exception mockException = Mockito.mock(Exception.class);
        // Set up the mock behavior
        when(mockException.getMessage()).thenReturn("Mocked Exception");
        // Act
        String result = obj.ioError(mockException);
        // Assert
        assertEquals("IO Error: Mocked Exception", result);
    }
}
