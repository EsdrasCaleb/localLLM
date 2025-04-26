package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_error_0_0_Test {

    @Test
    public void testErrorMethodWithNullException() {
        // Given
        Exception ex = null;
        // When
        String result = AnyWrapperMsgGenerator.error(ex);
        // Then
        assertEquals("Error - null", result);
    }

    @Test
    public void testErrorMethodWithNonNullException() {
        // Given
        Exception ex = Mockito.mock(Exception.class);
        Mockito.when(ex.toString()).thenReturn("MockedExceptionMessage");
        // When
        String result = AnyWrapperMsgGenerator.error(ex);
        // Then
        assertEquals("Error - MockedExceptionMessage", result);
    }
}
