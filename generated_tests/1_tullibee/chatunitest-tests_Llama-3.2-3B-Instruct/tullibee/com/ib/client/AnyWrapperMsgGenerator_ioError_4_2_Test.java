package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class AnyWrapperMsgGenerator_ioError_4_2_Test {

    @Mock
    private Exception exception;

    @InjectMocks
    private AnyWrapperMsgGenerator anyWrapperMsgGenerator;

    @Test
    public void testIoError_GeneratesErrorMessage() {
        // Arrange
        when(exception.getMessage()).thenReturn("Mock Exception Message");
        // Act
        String errorMessage = AnyWrapperMsgGenerator.ioError(exception);
        // Assert
        assertEquals("Mock Exception Message", errorMessage);
    }

    @Test
    public void testIoError_ThrowsExceptionIfInputIsNotException() {
        // Arrange
        // Create an instance of Exception
        Exception nonExceptionInput = new Exception();
        // Act and Assert
        assertThrows(Exception.class, () -> AnyWrapperMsgGenerator.ioError(nonExceptionInput));
    }
}
