package com.ib.client;

import static org.junit.Assert.assertEquals;
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
        AnyWrapperMsgGenerator generator = new AnyWrapperMsgGenerator();
        // Act
        String result = generator.error("test");
        // Assert
        assertEquals("test", result);
    }
}
