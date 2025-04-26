package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class AnyWrapperMsgGenerator_connectionClosed_3_1_Test {

    @Test
    void testConnectionClosed() {
        // Arrange
        AnyWrapperMsgGenerator anyWrapperMsgGenerator = new AnyWrapperMsgGenerator();
        String expectedMessage = "Connection Closed";
        // Act
        String actualMessage = anyWrapperMsgGenerator.connectionClosed();
        // Assert
        assertEquals(expectedMessage, actualMessage);
    }
}
