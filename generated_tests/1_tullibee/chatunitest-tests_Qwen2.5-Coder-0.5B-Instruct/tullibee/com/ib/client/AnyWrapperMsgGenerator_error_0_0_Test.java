package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class AnyWrapperMsgGenerator_error_0_0_Test {

    @Test
    public void testError() throws Exception {
        // Arrange
        Exception exception = new Exception("Something went wrong");
        // Act
        String result = AnyWrapperMsgGenerator.error(exception);
        // Assert
        assertEquals("Error - Something went wrong", result);
    }
}
