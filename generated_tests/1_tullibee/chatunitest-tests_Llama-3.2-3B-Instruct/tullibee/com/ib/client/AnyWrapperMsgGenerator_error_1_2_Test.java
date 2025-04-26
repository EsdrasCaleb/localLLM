package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class AnyWrapperMsgGenerator_error_1_2_Test {

    @Test
    public void testError(String input) {
        // Arrange
        AnyWrapperMsgGenerator underTest = new AnyWrapperMsgGenerator();
        // Act
        String result = underTest.error(input);
        // Assert
        assertEquals(input, result);
    }
}
