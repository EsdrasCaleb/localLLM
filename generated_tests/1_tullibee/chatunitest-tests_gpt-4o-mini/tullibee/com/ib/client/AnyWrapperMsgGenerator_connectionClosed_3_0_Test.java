package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_connectionClosed_3_0_Test {

    @Test
    public void testConnectionClosed() {
        // Arrange
        String expectedMessage = "Connection Closed";
        // Act
        String actualMessage = AnyWrapperMsgGenerator.connectionClosed();
        // Assert
        assertEquals(expectedMessage, actualMessage);
    }
}
