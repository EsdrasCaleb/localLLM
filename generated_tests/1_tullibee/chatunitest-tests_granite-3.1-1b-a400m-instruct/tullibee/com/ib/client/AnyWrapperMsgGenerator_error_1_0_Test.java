package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_error_1_0_Test {

    @Test
    public void testError() {
        // Arrange
        String str = "Test message";
        AnyWrapperMsgGenerator generator = new AnyWrapperMsgGenerator();
        // Act
        String result = generator.error(str);
        // Assert
        assertEquals(str, result);
    }
}
