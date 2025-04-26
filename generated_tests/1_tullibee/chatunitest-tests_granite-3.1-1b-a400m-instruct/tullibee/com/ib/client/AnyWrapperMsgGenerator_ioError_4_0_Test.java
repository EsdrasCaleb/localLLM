package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_ioError_4_0_Test {

    @Test
    public void testIoError() {
        // Arrange
        AnyWrapperMsgGenerator generator = new AnyWrapperMsgGenerator();
        Exception exception = new RuntimeException("Test exception");
        // Act
        String errorMessage = generator.ioError(exception);
        // Assert
        assertEquals("Test exception", errorMessage);
    }
}
