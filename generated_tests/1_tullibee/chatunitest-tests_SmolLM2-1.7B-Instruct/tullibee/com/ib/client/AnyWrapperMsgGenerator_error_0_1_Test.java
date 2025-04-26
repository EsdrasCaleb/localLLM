package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class AnyWrapperMsgGenerator_error_0_1_Test {

    @Test
    public void testError() {
        // Arrange
        Exception ex = new Exception("Mocked Exception");
        // Act
        String result = AnyWrapperMsgGenerator.error(ex);
        // Assert
        Assertions.assertEquals("Error - Mocked Exception", result);
    }
}
