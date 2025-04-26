package com.ib.client;

import java.lang.Exception;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_error_0_0_Test {

    @Test
    public void testError() {
        // Given
        Exception ex = mock(Exception.class);
        String expected = "Error - " + ex;
        // When
        String result = AnyWrapperMsgGenerator.error(ex);
        // Then
        assertEquals(expected, result);
    }
}
